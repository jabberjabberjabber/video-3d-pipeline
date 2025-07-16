"""Depth extraction from SBS stereoscopic video using CREStereo."""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
import ffmpeg
from pathlib import Path
import hashlib
import os
import argparse
from typing import Tuple, List, Optional, Dict
import gc
import warnings
from collections import defaultdict

from .utils import get_video_info, create_work_directory


class HybridStereoDepthExtractor:
    """ GPU-accelerated depth extraction from SBS video using hybrid stereo matching + neural guidance """
    
    def __init__(self, 
                 model_checkpoint: str = "Intel/dpt-large",
                 work_dir: str = "temp_depth",
                 cache_dir: str = "temp_depth",
                 device: str = "cuda",
                 batch_size: int = 8,
                 use_neural_guidance: bool = True,
                 stereo_only: bool = False):
        
        self.device = device
        self.work_dir = create_work_directory(work_dir)
        self.cache_dir = create_work_directory(cache_dir)
        self.batch_size = batch_size
        self.model_checkpoint = model_checkpoint
        self.use_neural_guidance = use_neural_guidance
        self.stereo_only = stereo_only
        
        # Verify CUDA availability
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA not available but requested")
        
        print(f"Initializing Hybrid Stereo depth extractor...")
        print(f"Device: {self.device}")
        print(f"Model: {self.model_checkpoint if not self.stereo_only else 'Stereo-only mode'}")
        print(f"Batch size: {self.batch_size}")
        print(f"Neural guidance: {self.use_neural_guidance and not self.stereo_only}")
        
        # Initialize model (will be loaded on first use)
        self.model = None
        self.model_loaded = False
        
        # Memory management
        self.max_vram_usage = 0.9  # Use 90% of available VRAM
        self.memory_stats = defaultdict(float)
    
    def load_model(self):
        """ Load depth estimation model to GPU """
        if self.model_loaded:
            return
        
        if self.stereo_only:
            print("Using stereo-only mode (no neural network)")
            self.model_loaded = True
            return
        
        print(f"Loading depth model: {self.model_checkpoint}")
        
        try:
            # Use DPT for monocular depth estimation as guidance
            from transformers import DPTImageProcessor, DPTForDepthEstimation
            
            print("Loading DPT model for neural depth guidance")
            self.processor = DPTImageProcessor.from_pretrained(self.model_checkpoint)
            self.model = DPTForDepthEstimation.from_pretrained(self.model_checkpoint)
            
            # Move to GPU
            if self.device == "cuda":
                self.model = self.model.to(self.device)
                self.model.eval()
                
                # Get GPU memory info
                total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
                print(f"GPU: {torch.cuda.get_device_name(0)}")
                print(f"Total VRAM: {total_memory:.1f}GB")
                
                # Check model memory usage
                torch.cuda.empty_cache()
                model_memory = torch.cuda.memory_allocated() / 1e9
                print(f"Model VRAM usage: {model_memory:.1f}GB")
                
                # Calculate optimal batch size based on remaining memory
                available_memory = (total_memory * self.max_vram_usage) - model_memory
                estimated_per_frame = 0.8  # GB per 1080p frame pair (rough estimate)
                optimal_batch = max(1, int(available_memory / estimated_per_frame))
                
                if optimal_batch < self.batch_size:
                    print(f"Reducing batch size from {self.batch_size} to {optimal_batch} for memory")
                    self.batch_size = optimal_batch
                
            self.model_loaded = True
            print("✓ Model loaded successfully")
            
        except ImportError:
            print("Warning: transformers library not available, falling back to stereo-only mode")
            self.stereo_only = True
            self.model_loaded = True
        except Exception as e:
            print(f"Warning: Failed to load neural model, falling back to stereo-only mode: {e}")
            self.stereo_only = True
            self.model_loaded = True
    
    def get_cache_path(self, video_path: str, frame_start: int, frame_count: int) -> Path:
        """ Generate cache path for depth maps """
        # Create cache key from video path and frame range
        cache_key = f"{video_path}_{frame_start}_{frame_count}_{self.model_checkpoint}"
        cache_hash = hashlib.md5(cache_key.encode()).hexdigest()[:16]
        
        cache_subdir = self.cache_dir / f"depth_{cache_hash}"
        cache_subdir.mkdir(exist_ok=True)
        
        return cache_subdir
    
    def is_cached(self, cache_path: Path, frame_count: int) -> bool:
        """ Check if depth maps are already cached """
        if not cache_path.exists():
            return False
        
        # Check if all expected depth maps exist
        expected_files = [cache_path / f"depth_{i:06d}.png" for i in range(frame_count)]
        all_exist = all(f.exists() for f in expected_files)
        
        if all_exist:
            print(f"✓ Found cached depth maps: {cache_path}")
            return True
        
        return False

    def extract_frames_opencv(self, video_path: str, start_frame: int = 0, max_frames: int = None) -> List[np.ndarray]:
        """ Extract video frames using OpenCV """
        print(f"Extracting frames from {video_path} using OpenCV...")

        # Get video info
        video_info = get_video_info(video_path)
        if not video_info:
            raise ValueError(f"Could not read video info: {video_path}")

        total_frames = video_info.get('frames', 0) or int(video_info['duration'] * video_info['fps'])

        if max_frames is None:
            max_frames = total_frames - start_frame
        else:
            max_frames = min(max_frames, total_frames - start_frame)

        print(f"Extracting {max_frames} frames starting from frame {start_frame}")

        frames = []

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

                frames.append(frame)
                frame_count += 1

                if frame_count % 100 == 0:
                    print(f"Extracted {frame_count}/{max_frames} frames...")

            cap.release()

        except Exception as e:
            raise RuntimeError(f"Frame extraction failed: {e}")

        print(f"✓ Extracted {len(frames)} frames")
        return frames
        
    def extract_frames_ffmpeg(self, video_path: str, start_frame: int = 0, max_frames: int = None) -> List[np.ndarray]:
        """ Extract video frames using ffmpeg """
        print(f"Extracting frames from {video_path}...")
        
        # Get video info
        video_info = get_video_info(video_path)
        if not video_info:
            raise ValueError(f"Could not read video info: {video_path}")
        
        total_frames = video_info.get('frames', 0) or int(video_info['duration'] * video_info['fps'])
        
        if max_frames is None:
            max_frames = total_frames - start_frame
        else:
            max_frames = min(max_frames, total_frames - start_frame)
        
        print(f"Extracting {max_frames} frames starting from frame {start_frame}")
        
        frames = []
        
        try:
            # Create ffmpeg process for frame extraction
            start_time = start_frame / video_info['fps']
            duration = max_frames / video_info['fps']
            
            process = (
                ffmpeg
                .input(video_path, ss=start_time, t=duration)
                .output('pipe:', format='rawvideo', pix_fmt='rgb24')
                .run_async(pipe_stdout=True, pipe_stderr=True, quiet=True)
            )
            
            frame_size = video_info['width'] * video_info['height'] * 3  # RGB
            
            frame_count = 0
            while frame_count < max_frames:
                # Read frame data
                frame_data = process.stdout.read(frame_size)
                if not frame_data or len(frame_data) != frame_size:
                    break
                
                # Convert to numpy array
                frame = np.frombuffer(frame_data, np.uint8).reshape(
                    (video_info['height'], video_info['width'], 3)
                )
                
                frames.append(frame)
                frame_count += 1
                
                if frame_count % 100 == 0:
                    print(f"Extracted {frame_count}/{max_frames} frames...")
            
            process.wait()
            
        except Exception as e:
            raise RuntimeError(f"Frame extraction failed: {e}")
        
        print(f"✓ Extracted {len(frames)} frames")
        return frames
    
    def split_sbs_frame(self, sbs_frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """ Split side-by-side frame into left and right images """
        height, width = sbs_frame.shape[:2]
        
        if width % 2 != 0:
            raise ValueError("SBS frame width must be even")
        
        half_width = width // 2
        left_frame = sbs_frame[:, :half_width]
        right_frame = sbs_frame[:, half_width:]
        
        return left_frame, right_frame
    
    def preprocess_frame_pair(self, left_frame: np.ndarray, right_frame: np.ndarray) -> Dict:
        """ Preprocess frame pair for depth estimation """
        # Convert BGR to RGB if needed (OpenCV default is BGR)
        if left_frame.shape[2] == 3:
            left_rgb = cv2.cvtColor(left_frame, cv2.COLOR_BGR2RGB)
            right_rgb = cv2.cvtColor(right_frame, cv2.COLOR_BGR2RGB)
        else:
            left_rgb = left_frame
            right_rgb = right_frame
        
        result = {'stereo_pair': {'left': left_rgb, 'right': right_rgb}}
        
        # Only process for neural guidance if needed
        if self.use_neural_guidance and not self.stereo_only and hasattr(self, 'processor'):
            try:
                inputs = self.processor(images=left_rgb, return_tensors="pt")
                
                # Move to device
                if self.device == "cuda":
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                result['dpt_inputs'] = inputs
            except Exception as e:
                print(f"Warning: Neural preprocessing failed: {e}")
        
        return result
    
    def process_frame_batch(self, frame_pairs: List[Tuple[np.ndarray, np.ndarray]]) -> List[np.ndarray]:
        """ Process batch of frame pairs for depth estimation """
        if not self.model_loaded:
            self.load_model()
        
        batch_size = len(frame_pairs)
        print(f"Processing batch of {batch_size} frame pairs...")
        
        # Check GPU memory before processing
        if self.device == "cuda":
            torch.cuda.empty_cache()
            memory_before = torch.cuda.memory_allocated() / 1e9
            print(f"GPU memory before batch: {memory_before:.1f}GB")
        
        depth_maps = []
        
        try:
            # Create stereo matcher for traditional stereo matching
            stereo = cv2.StereoSGBM_create(
                minDisparity=0,
                numDisparities=64,  # Must be divisible by 16
                blockSize=5,
                P1=8 * 3 * 5**2,
                P2=32 * 3 * 5**2,
                disp12MaxDiff=1,
                uniquenessRatio=10,
                speckleWindowSize=100,
                speckleRange=32
            )
            
            with torch.no_grad():
                # Process each pair
                for i, (left, right) in enumerate(frame_pairs):
                    print(f"  Processing pair {i+1}/{batch_size}...")
                    
                    # Preprocess frames
                    processed = self.preprocess_frame_pair(left, right)
                    stereo_pair = processed['stereo_pair']
                    
                    # Convert to grayscale for stereo matching
                    left_gray = cv2.cvtColor(stereo_pair['left'], cv2.COLOR_RGB2GRAY)
                    right_gray = cv2.cvtColor(stereo_pair['right'], cv2.COLOR_RGB2GRAY)
                    
                    # Compute disparity using stereo matching
                    disparity = stereo.compute(left_gray, right_gray).astype(np.float32) / 16.0
                    
                    # Optional: Use neural guidance if enabled and available
                    if (self.use_neural_guidance and not self.stereo_only and 
                        hasattr(self, 'model') and 'dpt_inputs' in processed):
                        try:
                            dpt_inputs = processed['dpt_inputs']
                            dpt_outputs = self.model(**dpt_inputs)
                            monocular_depth = dpt_outputs.predicted_depth[0].cpu().numpy()
                            
                            # Resize monocular depth to match disparity
                            if monocular_depth.shape != disparity.shape:
                                monocular_depth = cv2.resize(monocular_depth, 
                                                            (disparity.shape[1], disparity.shape[0]))
                            
                            # Combine stereo disparity with monocular depth (weighted average)
                            # Normalize monocular depth to disparity range
                            if monocular_depth.max() > monocular_depth.min():
                                mono_normalized = ((monocular_depth - monocular_depth.min()) / 
                                                 (monocular_depth.max() - monocular_depth.min()) * 64)
                                
                                # Weighted combination (favor stereo for accuracy)
                                combined_disparity = 0.7 * disparity + 0.3 * mono_normalized
                            else:
                                combined_disparity = disparity
                                
                        except Exception as e:
                            print(f"    Warning: Neural guidance failed, using stereo only: {e}")
                            combined_disparity = disparity
                    else:
                        combined_disparity = disparity
                    
                    # Clean up disparity (remove invalid values)
                    combined_disparity[combined_disparity <= 0] = 0
                    
                    depth_maps.append(combined_disparity.astype(np.float32))
                
        except torch.cuda.OutOfMemoryError as e:
            print(f"GPU memory error in batch processing: {e}")
            print("Try reducing batch size")
            torch.cuda.empty_cache()
            raise
        except Exception as e:
            print(f"Error processing frame batch: {e}")
            raise
        
        finally:
            # Clean up GPU memory
            if self.device == "cuda":
                torch.cuda.empty_cache()
                memory_after = torch.cuda.memory_allocated() / 1e9
                print(f"GPU memory after batch: {memory_after:.1f}GB")
        
        print(f"✓ Processed {len(depth_maps)} depth maps")
        return depth_maps
    
    def save_depth_map(self, depth_map: np.ndarray, output_path: Path):
        """ Save depth map as 16-bit PNG """
        # Normalize to 16-bit range
        if depth_map.max() > depth_map.min():
            normalized = ((depth_map - depth_map.min()) / (depth_map.max() - depth_map.min()) * 65535).astype(np.uint16)
        else:
            normalized = np.zeros_like(depth_map, dtype=np.uint16)
        
        # Save as PNG
        cv2.imwrite(str(output_path), normalized)
    
    def process_video_sbs(self, 
                         video_path: str, 
                         start_frame: int = 0, 
                         max_frames: int = None,
                         force_reprocess: bool = False) -> Path:
        """ Process entire SBS video to extract depth maps """
        
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
        
        # Check cache
        cache_path = self.get_cache_path(video_path, start_frame, frame_count)
        
        if not force_reprocess and self.is_cached(cache_path, frame_count):
            print("✓ Using cached depth maps")
            return cache_path
        
        # Extract frames
        frames = self.extract_frames_opencv(video_path, start_frame, frame_count)
        
        if not frames:
            raise ValueError("No frames extracted from video")
        
        # Process frames in batches
        processed_count = 0
        
        for batch_start in range(0, len(frames), self.batch_size):
            batch_end = min(batch_start + self.batch_size, len(frames))
            batch_frames = frames[batch_start:batch_end]
            
            print(f"Processing batch {batch_start//self.batch_size + 1}: frames {batch_start}-{batch_end-1}")
            
            # Split SBS frames into left/right pairs
            frame_pairs = []
            for frame in batch_frames:
                left, right = self.split_sbs_frame(frame)
                frame_pairs.append((left, right))
            
            # Process batch
            depth_maps = self.process_frame_batch(frame_pairs)
            
            # Save depth maps
            for i, depth_map in enumerate(depth_maps):
                frame_idx = batch_start + i
                output_path = cache_path / f"depth_{frame_idx:06d}.png"
                self.save_depth_map(depth_map, output_path)
                processed_count += 1
            
            print(f"✓ Saved batch depth maps ({processed_count}/{len(frames)} total)")
        
        print(f"✓ Depth extraction complete: {cache_path}")
        print(f"  Processed {processed_count} frames")
        print(f"  Output directory: {cache_path}")
        
        return cache_path


def main():
    """ Command line interface for depth extraction """
    parser = argparse.ArgumentParser(description='Extract depth maps from SBS stereoscopic video')
    parser.add_argument('video', help='Path to SBS video file')
    parser.add_argument('--start-frame', type=int, default=0,
                       help='Starting frame number (default: 0)')
    parser.add_argument('--max-frames', type=int, default=None,
                       help='Maximum number of frames to process (default: all)')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Batch size for GPU processing (default: 8)')
    parser.add_argument('--model', default="Intel/dpt-large",
                       help='Neural model checkpoint (default: Intel/dpt-large)')
    parser.add_argument('--work-dir', default='temp_depth',
                       help='Working directory for output (default: temp_depth)')
    parser.add_argument('--force', action='store_true',
                       help='Force reprocessing even if cached results exist')
    parser.add_argument('--device', default='cuda',
                       help='Processing device (default: cuda)')
    parser.add_argument('--stereo-only', action='store_true',
                       help='Use stereo matching only (no neural guidance)')
    parser.add_argument('--no-neural', action='store_true',
                       help='Disable neural guidance (same as --stereo-only)')
    
    args = parser.parse_args()
    
    # Handle neural guidance flags
    stereo_only = args.stereo_only or args.no_neural
    use_neural_guidance = not stereo_only
    
    try:
        # Initialize depth extractor
        extractor = HybridStereoDepthExtractor(
            model_checkpoint=args.model,
            work_dir=args.work_dir,
            cache_dir=args.work_dir,
            device=args.device,
            batch_size=args.batch_size,
            use_neural_guidance=use_neural_guidance,
            stereo_only=stereo_only
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
