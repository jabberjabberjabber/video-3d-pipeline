"""Depth map upscaling from 1080p to 4K using guided interpolation."""

import cv2
import numpy as np
import torch
import ffmpeg
from pathlib import Path
import hashlib
import os
import argparse
from typing import List, Tuple, Optional
import gc
from tqdm import tqdm
import warnings

from .utils import get_video_info, create_work_directory


class GuidedDepthUpscaler:
    """ GPU-accelerated depth upscaling using 4K 2D frames as guidance """
    
    def __init__(self, 
                 work_dir: str = "temp_upscale",
                 cache_dir: str = "temp_upscale", 
                 device: str = "cuda",
                 upscale_method: str = "simple",
                 guided_filter_radius: int = 8,
                 guided_filter_eps: float = 0.01):
        
        self.device = device
        self.work_dir = create_work_directory(work_dir)
        self.cache_dir = create_work_directory(cache_dir)
        self.upscale_method = upscale_method
        self.guided_filter_radius = guided_filter_radius
        self.guided_filter_eps = guided_filter_eps
        
        # Verify CUDA availability
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA not available but requested")
        
        print(f"Initializing Depth Upscaler...")
        print(f"Device: {self.device}")
        print(f"Upscale method: {self.upscale_method}")
        if self.upscale_method == "guided":
            print(f"Guided filter radius: {self.guided_filter_radius}")
            print(f"Guided filter epsilon: {self.guided_filter_eps}")
        
        # Load CUDA kernels for guided filter if available
        self.cuda_available = device == "cuda" and torch.cuda.is_available()
        if self.cuda_available:
            print("✓ CUDA acceleration available")
    
    def get_cache_key(self, depth_dir: str, video_4k_path: str, method: str) -> str:
        """ Generate cache key for upscaled depth video """
        cache_string = f"{depth_dir}_{video_4k_path}_{method}_{self.guided_filter_radius}_{self.guided_filter_eps}"
        return hashlib.md5(cache_string.encode()).hexdigest()[:16]
    
    def load_depth_maps(self, depth_dir: Path) -> List[np.ndarray]:
        """ Load depth maps from directory """
        print(f"Loading depth maps from {depth_dir}...")
        
        # Find all depth map files
        depth_files = sorted(depth_dir.glob("depth_*.png"))
        
        if not depth_files:
            raise ValueError(f"No depth maps found in {depth_dir}")
        
        print(f"Found {len(depth_files)} depth maps")
        
        depth_maps = []
        for depth_file in tqdm(depth_files, desc="Loading depth maps"):
            # Load as 16-bit grayscale
            depth_map = cv2.imread(str(depth_file), cv2.IMREAD_ANYDEPTH)
            if depth_map is None:
                raise ValueError(f"Could not load depth map: {depth_file}")
            
            # Convert to float32 and normalize
            depth_map = depth_map.astype(np.float32) / 65535.0
            depth_maps.append(depth_map)
        
        print(f"✓ Loaded {len(depth_maps)} depth maps")
        return depth_maps
    
    def extract_4k_frames(self, video_path: str, max_frames: int = None) -> List[np.ndarray]:
        """ Extract 4K frames using OpenCV """
        print(f"Extracting 4K frames from {video_path}...")
        
        # Get video info
        video_info = get_video_info(video_path)
        if not video_info:
            raise ValueError(f"Could not read video info: {video_path}")
        
        total_frames = video_info.get('frames', 0) or int(video_info['duration'] * video_info['fps'])
        
        if max_frames is None:
            max_frames = total_frames
        else:
            max_frames = min(max_frames, total_frames)
        
        print(f"Extracting {max_frames} frames at {video_info['width']}x{video_info['height']}")
        
        frames = []
        
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Could not open video file: {video_path}")
            
            frame_count = 0
            pbar = tqdm(total=max_frames, desc="Extracting 4K frames")
            
            while frame_count < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frames.append(frame)
                frame_count += 1
                pbar.update(1)
            
            cap.release()
            pbar.close()
            
        except Exception as e:
            raise RuntimeError(f"Frame extraction failed: {e}")
        
        print(f"✓ Extracted {len(frames)} 4K frames")
        return frames
    
    def guided_filter_cuda(self, guide: np.ndarray, src: np.ndarray, radius: int, eps: float) -> np.ndarray:
        """ CUDA-accelerated guided filter using PyTorch """
        if not self.cuda_available:
            return self.guided_filter_cpu(guide, src, radius, eps)
        
        try:
            # Convert to PyTorch tensors
            guide_tensor = torch.from_numpy(guide).float().cuda()
            src_tensor = torch.from_numpy(src).float().cuda()
            
            # Add batch and channel dimensions
            if len(guide_tensor.shape) == 3:
                guide_tensor = guide_tensor.unsqueeze(0).permute(0, 3, 1, 2)  # BHWC -> BCHW
            else:
                guide_tensor = guide_tensor.unsqueeze(0).unsqueeze(0)  # HW -> BCHW
            
            if len(src_tensor.shape) == 2:
                src_tensor = src_tensor.unsqueeze(0).unsqueeze(0)  # HW -> BCHW
            
            # Use PyTorch's built-in convolution for box filter approximation
            kernel_size = radius * 2 + 1
            kernel = torch.ones(1, 1, kernel_size, kernel_size, device='cuda') / (kernel_size * kernel_size)
            
            # Box filter function
            def box_filter(x):
                return torch.nn.functional.conv2d(x, kernel, padding=radius)
            
            # Guided filter computation
            N = box_filter(torch.ones_like(guide_tensor[:, :1]))
            mean_I = box_filter(guide_tensor) / N
            mean_p = box_filter(src_tensor) / N
            mean_Ip = box_filter(guide_tensor * src_tensor) / N
            cov_Ip = mean_Ip - mean_I * mean_p
            
            mean_II = box_filter(guide_tensor * guide_tensor) / N
            var_I = mean_II - mean_I * mean_I
            
            a = cov_Ip / (var_I + eps)
            b = mean_p - a * mean_I
            
            mean_a = box_filter(a) / N
            mean_b = box_filter(b) / N
            
            result = mean_a * guide_tensor + mean_b
            
            # Convert back to numpy
            result = result.squeeze().cpu().numpy()
            
            return result
            
        except Exception as e:
            print(f"Warning: CUDA guided filter failed, falling back to CPU: {e}")
            return self.guided_filter_cpu(guide, src, radius, eps)
    
    def guided_filter_cpu(self, guide: np.ndarray, src: np.ndarray, radius: int, eps: float) -> np.ndarray:
        """ CPU-based guided filter implementation """
        # Convert guide to grayscale if needed
        if len(guide.shape) == 3:
            guide_gray = cv2.cvtColor(guide, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        else:
            guide_gray = guide.astype(np.float32)
        
        # Ensure src is float32
        src = src.astype(np.float32)
        
        # Box filter kernel
        kernel_size = radius * 2 + 1
        kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
        
        # Box filter function
        def box_filter(x):
            return cv2.filter2D(x, -1, kernel)
        
        # Guided filter computation
        mean_I = box_filter(guide_gray)
        mean_p = box_filter(src)
        mean_Ip = box_filter(guide_gray * src)
        cov_Ip = mean_Ip - mean_I * mean_p
        
        mean_II = box_filter(guide_gray * guide_gray)
        var_I = mean_II - mean_I * mean_I
        
        a = cov_Ip / (var_I + eps)
        b = mean_p - a * mean_I
        
        mean_a = box_filter(a)
        mean_b = box_filter(b)
        
        result = mean_a * guide_gray + mean_b
        
        return result
    
    def upscale_depth_simple(self, depth_map: np.ndarray, target_height: int, target_width: int) -> np.ndarray:
        """ Simple bicubic upscaling without guidance """
        # Direct bicubic resize
        depth_upscaled = cv2.resize(depth_map, (target_width, target_height), interpolation=cv2.INTER_CUBIC)
        
        # Ensure proper range
        depth_upscaled = np.clip(depth_upscaled, 0, 1)
        
        return depth_upscaled
    
    def upscale_depth_guided(self, depth_map: np.ndarray, guide_frame: np.ndarray) -> np.ndarray:
        """ Upscale depth map using guided filtering """
        
        # Get target dimensions from guide frame
        target_height, target_width = guide_frame.shape[:2]
        
        # Initial bicubic upscaling
        depth_upscaled = cv2.resize(depth_map, (target_width, target_height), interpolation=cv2.INTER_CUBIC)
        
        # Apply guided filter for edge-preserving refinement
        if self.upscale_method == "guided":
            depth_refined = self.guided_filter_cuda(
                guide_frame, 
                depth_upscaled, 
                self.guided_filter_radius, 
                self.guided_filter_eps
            )
        else:
            depth_refined = depth_upscaled
        
        # Ensure proper range
        depth_refined = np.clip(depth_refined, 0, 1)
        
        return depth_refined
    
    def upscale_depth_batch(self, depth_maps: List[np.ndarray], guide_frames: List[np.ndarray] = None, target_height: int = None, target_width: int = None) -> List[np.ndarray]:
        """ Upscale batch of depth maps """
        
        if self.upscale_method == "simple" or self.upscale_method == "bicubic":
            # Simple upscaling doesn't need guide frames
            if target_height is None or target_width is None:
                if guide_frames is None or len(guide_frames) == 0:
                    raise ValueError("For simple upscaling, either provide guide_frames or target dimensions")
                target_height, target_width = guide_frames[0].shape[:2]
            
            print(f"Simple upscaling {len(depth_maps)} depth maps to {target_width}x{target_height}...")
            
            upscaled_depths = []
            for i, depth_map in enumerate(tqdm(depth_maps, desc="Upscaling depth maps (simple)")):
                upscaled_depth = self.upscale_depth_simple(depth_map, target_height, target_width)
                upscaled_depths.append(upscaled_depth)
                
                # Memory cleanup
                if i % 100 == 0:
                    import gc
                    gc.collect()
            
        else:
            # Guided upscaling needs guide frames
            if guide_frames is None:
                raise ValueError("Guided upscaling requires guide frames")
            
            if len(depth_maps) != len(guide_frames):
                raise ValueError(f"Mismatch: {len(depth_maps)} depth maps vs {len(guide_frames)} guide frames")
            
            print(f"Guided upscaling {len(depth_maps)} depth maps...")
            
            upscaled_depths = []
            for i, (depth_map, guide_frame) in enumerate(tqdm(zip(depth_maps, guide_frames), 
                                                             desc="Upscaling depth maps (guided)", 
                                                             total=len(depth_maps))):
                
                upscaled_depth = self.upscale_depth_guided(depth_map, guide_frame)
                upscaled_depths.append(upscaled_depth)
                
                # Memory cleanup
                if i % 50 == 0 and self.cuda_available:
                    torch.cuda.empty_cache()
        
        print(f"✓ Upscaled {len(upscaled_depths)} depth maps")
        return upscaled_depths
    
    def save_depth_video(self, depth_maps: List[np.ndarray], output_path: Path, fps: float = 23.976):
        """ Save depth maps as video file """
        print(f"Saving depth video to {output_path}...")
        
        if not depth_maps:
            raise ValueError("No depth maps to save")
        
        # Get dimensions
        height, width = depth_maps[0].shape[:2]
        
        # Convert depth maps to 8-bit for video encoding
        depth_frames_8bit = []
        for depth_map in depth_maps:
            # Convert to 8-bit grayscale
            depth_8bit = (depth_map * 255).astype(np.uint8)
            depth_frames_8bit.append(depth_8bit)
        
        try:
            # Create FFmpeg process for video encoding
            process = (
                ffmpeg
                .input('pipe:', format='rawvideo', pix_fmt='gray', s=f'{width}x{height}', r=fps)
                .output(str(output_path), 
                       vcodec='libx264',
                       pix_fmt='yuv420p',
                       crf=18,
                       preset='medium')
                .overwrite_output()
                .run_async(pipe_stdin=True, pipe_stderr=True, quiet=True)
            )
            
            # Write frames
            for depth_frame in tqdm(depth_frames_8bit, desc="Encoding depth video"):
                process.stdin.write(depth_frame.tobytes())
            
            process.stdin.close()
            process.wait()
            
            if process.returncode != 0:
                stderr = process.stderr.read().decode() if process.stderr else ""
                raise RuntimeError(f"FFmpeg encoding failed: {stderr}")
            
        except Exception as e:
            raise RuntimeError(f"Video encoding failed: {e}")
        
        print(f"✓ Depth video saved: {output_path}")
    
    def process_depth_upscaling(self, 
                               depth_dir: str,
                               video_4k_path: str,
                               output_path: str = None,
                               force_reprocess: bool = False) -> str:
        """ Main pipeline for depth upscaling """
        
        print(f"Processing depth upscaling...")
        print(f"Depth maps: {depth_dir}")
        print(f"4K video: {video_4k_path}")
        
        # Generate cache key and output path
        cache_key = self.get_cache_key(depth_dir, video_4k_path, self.upscale_method)
        
        if output_path is None:
            output_path = self.work_dir / f"depth_4k_{cache_key}.mp4"
        else:
            output_path = Path(output_path)
        
        # Check if already processed
        if output_path.exists() and not force_reprocess:
            print(f"✓ Using cached upscaled depth video: {output_path}")
            return str(output_path)
        
        # Load depth maps
        depth_maps = self.load_depth_maps(Path(depth_dir))
        
        # Get target dimensions from 4K video info
        video_info = get_video_info(video_4k_path)
        if not video_info:
            raise ValueError(f"Could not read video info: {video_4k_path}")
        
        target_width = video_info['width']
        target_height = video_info['height']
        fps = video_info['fps']
        
        print(f"Target resolution: {target_width}x{target_height}")
        
        # Extract guide frames only if needed for guided upscaling
        if self.upscale_method == "guided":
            print("Extracting 4K frames for guided upscaling...")
            guide_frames = self.extract_4k_frames(video_4k_path, max_frames=len(depth_maps))
            
            # Verify frame count match
            if len(depth_maps) != len(guide_frames):
                min_frames = min(len(depth_maps), len(guide_frames))
                print(f"Warning: Frame count mismatch, using first {min_frames} frames")
                depth_maps = depth_maps[:min_frames]
                guide_frames = guide_frames[:min_frames]
            
            # Upscale depth maps with guidance
            upscaled_depths = self.upscale_depth_batch(depth_maps, guide_frames)
        else:
            print("Using simple bicubic upscaling (no guidance needed)...")
            # Simple upscaling doesn't need guide frames
            upscaled_depths = self.upscale_depth_batch(
                depth_maps, 
                target_height=target_height, 
                target_width=target_width
            )
        
        # Save as video
        self.save_depth_video(upscaled_depths, output_path, fps)
        
        print(f"✓ Depth upscaling complete!")
        print(f"  Input: {len(depth_maps)} depth maps @ {depth_maps[0].shape}")
        print(f"  Output: {output_path}")
        print(f"  Resolution: {upscaled_depths[0].shape}")
        
        return str(output_path)


def main():
    """ Command line interface for depth upscaling """
    parser = argparse.ArgumentParser(description='Upscale depth maps to 4K using guided filtering')
    parser.add_argument('depth_dir', help='Directory containing 1080p depth maps')
    parser.add_argument('video_4k', help='Path to 4K 2D video for guidance')
    parser.add_argument('--output', help='Output path for 4K depth video')
    parser.add_argument('--method', default='simple', choices=['guided', 'bicubic', 'simple'],
                       help='Upscaling method (default: simple)')
    parser.add_argument('--radius', type=int, default=8,
                       help='Guided filter radius (default: 8)')
    parser.add_argument('--eps', type=float, default=0.01,
                       help='Guided filter epsilon (default: 0.01)')
    parser.add_argument('--work-dir', default='temp_upscale',
                       help='Working directory (default: temp_upscale)')
    parser.add_argument('--force', action='store_true',
                       help='Force reprocessing even if output exists')
    parser.add_argument('--device', default='cuda',
                       help='Processing device (default: cuda)')
    
    args = parser.parse_args()
    
    try:
        # Initialize upscaler
        upscaler = GuidedDepthUpscaler(
            work_dir=args.work_dir,
            cache_dir=args.work_dir,
            device=args.device,
            upscale_method=args.method,
            guided_filter_radius=args.radius,
            guided_filter_eps=args.eps
        )
        
        # Process upscaling
        output_path = upscaler.process_depth_upscaling(
            depth_dir=args.depth_dir,
            video_4k_path=args.video_4k,
            output_path=args.output,
            force_reprocess=args.force
        )
        
        print(f"\n✓ Success! 4K depth video saved to: {output_path}")
        print(f"Ready for VisionDepth3D processing!")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
