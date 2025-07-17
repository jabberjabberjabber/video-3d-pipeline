"""Depth extraction from SBS stereoscopic video using IGEV stereo."""

import sys
import os
from pathlib import Path

# Add IGEV to path
project_root = Path(__file__).parent.parent.parent
igev_path = project_root / "IGEV"
sys.path.append(str(igev_path))
sys.path.append(str(igev_path / "core"))

import torch
import torch.nn.functional as F
import numpy as np
import cv2
import ffmpeg
import hashlib
import argparse
from typing import Tuple, List, Optional, Dict
import gc
import warnings
from collections import defaultdict
from tqdm import tqdm
import skimage.io

from core.igev_stereo import IGEVStereo
from core.utils.utils import InputPadder

from .utils import get_video_info, create_work_directory


class IGEVStereoDepthExtractor:
    """ GPU-accelerated depth extraction from SBS video using IGEV stereo """
    
    def __init__(self, 
                 model_checkpoint: str = "./models/sceneflow.pth",
                 work_dir: str = "temp_depth",
                 cache_dir: str = "temp_depth",
                 device: str = "cuda",
                 batch_size: int =16,
                 unsqueeze_sbs: bool = True,
                 valid_iters: int = 32):
        
        self.device = device
        self.work_dir = create_work_directory(work_dir)
        self.cache_dir = create_work_directory(cache_dir)
        self.batch_size = batch_size
        self.unsqueeze_sbs = unsqueeze_sbs
        self.valid_iters = valid_iters
        
        # Verify CUDA availability
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA not available but requested")
        
        # Set model checkpoint path
        if model_checkpoint is None:
            # Default to one of the available models
            models_dir = project_root / "models" / "Selective-IGEV"
            if (models_dir / "sceneflow").exists():
                self.model_checkpoint = str(models_dir / "sceneflow")
            elif (models_dir / "eth3d").exists():
                self.model_checkpoint = str(models_dir / "eth3d")
            else:
                raise ValueError(f"No IGEV models found in {models_dir}")
        else:
            self.model_checkpoint = model_checkpoint
        
        print(f"Initializing IGEV Stereo depth extractor...")
        print(f"Device: {self.device}")
        print(f"Model: {self.model_checkpoint}")
        print(f"Batch size: {self.batch_size}")
        print(f"Valid iterations: {self.valid_iters}")
        
        # Initialize model args (will be loaded with model)
        self.model = None
        self.model_loaded = False
        
        # Memory management
        self.max_vram_usage = 0.9  # Use 90% of available VRAM
        self.memory_stats = defaultdict(float)
    
    def load_model(self):
        """ Load IGEV stereo model to GPU """
        if self.model_loaded:
            return
        
        print(f"Loading IGEV model from {self.model_checkpoint}...")
        
        # Create args object with default IGEV parameters
        class Args:
            def __init__(self):
                self.mixed_precision = True
                self.precision_dtype = "float16"
                self.hidden_dims = [128] * 3
                self.corr_implementation = "reg"
                self.shared_backbone = False
                self.corr_levels = 2
                self.corr_radius = 4
                self.n_downsample = 2
                self.slow_fast_gru = False
                self.n_gru_layers = 3
                self.max_disp = 192
        
        args = Args()
        
        # Initialize model
        self.model = torch.nn.DataParallel(IGEVStereo(args), device_ids=[0])
        
        # Load checkpoint
        checkpoint_path = Path(self.model_checkpoint)
        if checkpoint_path.is_dir():
            # Look for .pth files in directory
            pth_files = list(checkpoint_path.glob("*.pth"))
            if not pth_files:
                raise ValueError(f"No .pth checkpoint files found in {checkpoint_path}")
            checkpoint_path = pth_files[0]  # Use first .pth file found
        
        if not checkpoint_path.exists():
            raise ValueError(f"Checkpoint not found: {checkpoint_path}")
        
        try:
            self.model.load_state_dict(torch.load(str(checkpoint_path), map_location=self.device))
            print(f"✓ Loaded checkpoint: {checkpoint_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load checkpoint {checkpoint_path}: {e}")
        
        self.model = self.model.module
        self.model.to(self.device)
        self.model.eval()
        
        self.model_loaded = True
        print("✓ IGEV model loaded successfully")
    
    def get_cache_path(self, video_path: str, frame_start: int, frame_count: int) -> Path:
        """ Generate cache path for depth maps """
        # Create cache key from video path and frame range
        cache_key = f"{video_path}_{frame_start}_{frame_count}_{self.model_checkpoint}_{self.unsqueeze_sbs}"
        cache_hash = hashlib.md5(cache_key.encode()).hexdigest()[:16]
        
        cache_subdir = self.cache_dir / f"depth_{cache_hash}"
        cache_subdir.mkdir(exist_ok=True)
        
        return cache_subdir
    
    def is_cached(self, cache_path: Path, frame_count: int) -> bool:
        """ Check if depth video is already cached (legacy method for compatibility) """
        return self.is_video_cached(cache_path)

    def get_frame_generator(self, video_path: str, start_frame: int = 0, max_frames: int = None):
        """ Generator that yields frames one at a time to avoid memory issues """
        print(f"Setting up frame generator for {video_path}...")

        # Get video info
        video_info = get_video_info(video_path)
        if not video_info:
            raise ValueError(f"Could not read video info: {video_path}")

        total_frames = video_info.get('frames', 0) or int(video_info['duration'] * video_info['fps'])

        if max_frames is None:
            max_frames = total_frames - start_frame
        else:
            max_frames = min(max_frames, total_frames - start_frame)

        print(f"Will process {max_frames} frames starting from frame {start_frame}")

        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Could not open video file: {video_path}")

            # Set start frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            frame_count = 0
            while frame_count < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break

                yield frame, frame_count
                frame_count += 1

                if frame_count % 100 == 0:
                    print(f"Processed {frame_count}/{max_frames} frames...")

            cap.release()
            print(f"✓ Completed processing {frame_count} frames")

        except Exception as e:
            cap.release()
            raise RuntimeError(f"Frame extraction failed: {e}")
        
    def split_sbs_frame(self, sbs_frame: np.ndarray, unsqueeze: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """ Split side-by-side frame into left and right images """
        height, width = sbs_frame.shape[:2]
        
        if width % 2 != 0:
            raise ValueError("SBS frame width must be even")
        
        half_width = width // 2
        left_frame = sbs_frame[:, :half_width]
        right_frame = sbs_frame[:, half_width:]
        
        # Unsqueeze horizontally to restore proper aspect ratio
        # SBS typically compresses each eye horizontally by 50%
        if unsqueeze:
            target_width = half_width * 2  # Restore original width
            left_frame = cv2.resize(left_frame, (target_width, height), interpolation=cv2.INTER_LANCZOS4)
            right_frame = cv2.resize(right_frame, (target_width, height), interpolation=cv2.INTER_LANCZOS4)
        
        return left_frame, right_frame
    
    def load_image_tensor(self, image: np.ndarray) -> torch.Tensor:
        """ Convert numpy image to tensor for IGEV """
        # Convert BGR to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert to tensor format expected by IGEV
        img_tensor = torch.from_numpy(image).permute(2, 0, 1).float()
        return img_tensor[None].to(self.device)
    
    def process_frame_batch(self, frame_pairs: List[Tuple[np.ndarray, np.ndarray]]) -> List[np.ndarray]:
        """ Process batch of frame pairs for depth estimation using IGEV """
        if not self.model_loaded:
            self.load_model()
        
        depth_maps = []
        
        with torch.no_grad():
            for left_frame, right_frame in tqdm(frame_pairs, desc="Processing frames"):
                try:
                    # Convert frames to tensors
                    left_tensor = self.load_image_tensor(left_frame)
                    right_tensor = self.load_image_tensor(right_frame)
                    
                    # Pad images to be divisible by 32
                    padder = InputPadder(left_tensor.shape, divis_by=32)
                    left_padded, right_padded = padder.pad(left_tensor, right_tensor)
                    
                    # Run IGEV stereo
                    disparity = self.model(left_padded, right_padded, 
                                         iters=self.valid_iters, test_mode=True)
                    
                    # Unpad result
                    disparity = padder.unpad(disparity)
                    
                    # Convert to numpy
                    disp_np = disparity.cpu().numpy().squeeze()
                    
                    # Convert disparity to depth (simple inverse relationship)
                    # You may want to adjust this based on your stereo setup
                    depth_map = np.where(disp_np > 0, 1.0 / (disp_np + 1e-6), 0)
                    
                    # Normalize depth to 0-1 range for saving
                    if depth_map.max() > depth_map.min():
                        depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
                    
                    depth_maps.append(depth_map)
                    
                except Exception as e:
                    print(f"Error processing frame pair: {e}")
                    # Create dummy depth map on error
                    h, w = left_frame.shape[:2]
                    depth_maps.append(np.zeros((h, w), dtype=np.float32))
                
                # Clean up GPU memory
                torch.cuda.empty_cache()
        
        return depth_maps
    
    def create_depth_video_writer(self, cache_path: Path, video_info: dict, output_height: int, output_width: int):
        """ Create FFmpeg video writer for depth maps """
        output_video_path = cache_path / "depth_video.mp4"
        
        # Use efficient encoding for depth data
        writer = (
            ffmpeg
            .input('pipe:', format='rawvideo', pix_fmt='gray16le', 
                   s=f"{output_width}x{output_height}", 
                   r=video_info['fps'])
            .output(str(output_video_path), 
                   vcodec='libx265',  # H.265 for better compression
                   crf=18,  # High quality, good compression
                   pix_fmt='yuv420p10le',  # 10-bit for depth precision
                   preset='fast')  # Reasonable encoding speed
            .overwrite_output()
            .run_async(pipe_stdin=True)
        )
        
        return writer, output_video_path
    
    def write_depth_to_video(self, depth_maps: List[np.ndarray], video_writer):
        """ Write batch of depth maps directly to video stream """
        for depth_map in depth_maps:
            # Convert to 16-bit for video encoding
            depth_16bit = np.round(depth_map * 65535).astype(np.uint16)
            
            # Write raw bytes to ffmpeg pipe
            video_writer.stdin.write(depth_16bit.tobytes())
    
    def is_video_cached(self, cache_path: Path) -> bool:
        """ Check if depth video is already cached """
        video_path = cache_path / "depth_video.mp4"
        return video_path.exists() and video_path.stat().st_size > 1000  # More than 1KB
    
    def process_video_sbs(self, 
                         video_path: str, 
                         start_frame: int = 0, 
                         max_frames: int = None,
                         force_reprocess: bool = False) -> Path:
        """ Process entire SBS video to extract depth maps using streaming to video """
        
        print(f"Processing SBS video: {video_path}")
        
        # Get video info
        video_info = get_video_info(video_path)
        if not video_info:
            raise ValueError(f"Could not read video info: {video_path}")
        
        total_frames = video_info.get('frames', 0) or int(video_info['duration'] * video_info['fps'])
        
        if max_frames is None:
            frame_count = total_frames - start_frame
        else:
            frame_count = min(max_frames, total_frames - start_frame)
        
        print(f"Video info: {video_info['width']}x{video_info['height']} @ {video_info['fps']:.1f}fps")
        print(f"Processing {frame_count} frames starting from frame {start_frame}")
        
        # Calculate output dimensions
        half_width = video_info['width'] // 2
        output_width = half_width * (2 if self.unsqueeze_sbs else 1)
        output_height = video_info['height']
        
        print(f"Output depth dimensions: {output_width}x{output_height}")
        
        # Check cache
        cache_path = self.get_cache_path(video_path, start_frame, frame_count)
        
        if not force_reprocess and self.is_video_cached(cache_path):
            print("✓ Using cached depth video")
            return cache_path / "depth_video.mp4"
        
        # Set up video writer for streaming depth maps
        video_writer = None
        output_video_path = None
        
        try:
            # Process frames in streaming batches to avoid memory issues
            processed_count = 0
            batch_frames = []
            batch_indices = []
            batch_num = 1
            
            # Get frame generator to avoid loading all frames into memory
            frame_generator = self.get_frame_generator(video_path, start_frame, frame_count)
            
            for frame, frame_idx in frame_generator:
                batch_frames.append(frame)
                batch_indices.append(frame_idx)
                
                # Process batch when it reaches the batch size or at the end
                if len(batch_frames) >= self.batch_size or frame_idx == frame_count - 1:
                    print(f"Processing batch {batch_num}: frames {batch_indices[0]}-{batch_indices[-1]}")
                    
                    # Initialize video writer on first batch
                    if video_writer is None:
                        video_writer, output_video_path = self.create_depth_video_writer(
                            cache_path, video_info, output_height, output_width
                        )
                        print(f"Started depth video encoding: {output_video_path}")
                    
                    # Split SBS frames into left/right pairs
                    frame_pairs = []
                    for frame in batch_frames:
                        left, right = self.split_sbs_frame(frame, unsqueeze=self.unsqueeze_sbs)
                        frame_pairs.append((left, right))
                    
                    # Process batch
                    depth_maps = self.process_frame_batch(frame_pairs)
                    
                    # Write depth maps directly to video stream
                    self.write_depth_to_video(depth_maps, video_writer)
                    processed_count += len(depth_maps)
                    
                    print(f"✓ Streamed batch to video ({processed_count}/{frame_count} total)")
                    
                    # Clear batch for next iteration
                    batch_frames = []
                    batch_indices = []
                    batch_num += 1
                    
                    # Force garbage collection to free memory
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            
            # Close video writer
            if video_writer:
                video_writer.stdin.close()
                video_writer.wait()
                
                # Verify output file was created successfully
                if output_video_path and output_video_path.exists():
                    file_size_mb = output_video_path.stat().st_size / (1024 * 1024)
                    print(f"✓ Depth video created: {output_video_path} ({file_size_mb:.1f}MB)")
                else:
                    raise RuntimeError("Video encoding failed - output file not created")
        
        except Exception as e:
            # Clean up video writer on error
            if video_writer:
                try:
                    video_writer.stdin.close()
                    video_writer.wait()
                except:
                    pass
            raise e
        
        print(f"✓ Depth extraction complete: {output_video_path}")
        print(f"  Processed {processed_count} frames")
        print(f"  Output: {output_video_path}")
        
        return output_video_path


def main():
    """ Command line interface for depth extraction """
    parser = argparse.ArgumentParser(description='Extract depth maps from SBS stereoscopic video using IGEV')
    parser.add_argument('video', help='Path to SBS video file')
    parser.add_argument('--start-frame', type=int, default=0,
                       help='Starting frame number (default: 0)')
    parser.add_argument('--max-frames', type=int, default=None,
                       help='Maximum number of frames to process (default: all)')
    parser.add_argument('--batch-size', type=int, default=4,
                       help='Batch size for GPU processing (default: 4)')
    parser.add_argument('--model', default=None,
                       help='IGEV model checkpoint path (default: auto-detect)')
    parser.add_argument('--work-dir', default='temp_depth',
                       help='Working directory for output (default: temp_depth)')
    parser.add_argument('--force', action='store_true',
                       help='Force reprocessing even if cached results exist')
    parser.add_argument('--device', default='cuda',
                       help='Processing device (default: cuda)')
    parser.add_argument('--no-unsqueeze', action='store_true',
                       help='Skip SBS unsqueezing (keep squeezed aspect ratio)')
    parser.add_argument('--valid-iters', type=int, default=32,
                       help='Number of IGEV refinement iterations (default: 32)')
    
    args = parser.parse_args()
    
    # Handle flags
    unsqueeze_sbs = not args.no_unsqueeze
    
    try:
        # Initialize depth extractor
        extractor = IGEVStereoDepthExtractor(
            model_checkpoint=args.model,
            work_dir=args.work_dir,
            cache_dir=args.work_dir,
            device=args.device,
            batch_size=args.batch_size,
            unsqueeze_sbs=unsqueeze_sbs,
            valid_iters=args.valid_iters
        )
        
        # Process video
        output_path = extractor.process_video_sbs(
            video_path=args.video,
            start_frame=args.start_frame,
            max_frames=args.max_frames,
            force_reprocess=args.force
        )
        
        print(f"\n✓ Success! Depth maps saved to: {output_path}")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
