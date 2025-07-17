"""Depth extraction from SBS stereoscopic video using UniMatch stereo."""

import sys
import os
from pathlib import Path

# Add UniMatch to path
project_root = Path(__file__).parent.parent.parent
unimatch_path = project_root / "unimatch"
sys.path.append(str(unimatch_path))

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

# Suppress the annoying low contrast warning
warnings.filterwarnings('ignore', message='.*is a low contrast image')

# UniMatch imports
from unimatch.unimatch import UniMatch
from utils.utils import InputPadder
from dataloader.stereo import transforms

from .utils import get_video_info, create_work_directory

# ImageNet normalization constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class UniMatchStereoDepthExtractor:
    """ GPU-accelerated depth extraction from SBS video using UniMatch stereo """
    
    def __init__(self, 
                 model_checkpoint: str = "./models/gmstereo-scale2-regrefine3-resumeflowthings-middleburyfthighres-a82bec03.pth",
                 work_dir: str = "temp_depth",
                 cache_dir: str = "temp_depth",
                 device: str = "cuda",
                 batch_size: int = 32,  # Increased for RTX 3080
                 unsqueeze_sbs: bool = True,
                 # UniMatch model parameters (optimized for speed)
                 num_scales: int = 1,  # Single scale for speed
                 feature_channels: int = 128,
                 upsample_factor: int = 4,
                 num_head: int = 1,
                 ffn_dim_expansion: int = 4,
                 num_transformer_layers: int = 6,
                 reg_refine: bool = True,
                 num_reg_refine: int = 1,  # Reduced for speed
                 attn_type: str = "self_swin2d_cross_1d",  # Faster attention
                 attn_splits_list: List[int] = None,
                 corr_radius_list: List[int] = None,
                 prop_radius_list: List[int] = None,
                 padding_factor: int = 32,
                 inference_size: Optional[List[int]] = None):
        
        self.device = device
        self.work_dir = create_work_directory(work_dir)
        self.cache_dir = create_work_directory(cache_dir)
        self.batch_size = batch_size
        self.unsqueeze_sbs = unsqueeze_sbs
        self.model_checkpoint = model_checkpoint
        self.padding_factor = padding_factor
        self.inference_size = inference_size
        
        # Verify CUDA availability
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA not available but requested")
        
        # UniMatch model parameters
        self.num_scales = num_scales
        self.feature_channels = feature_channels
        self.upsample_factor = upsample_factor
        self.num_head = num_head
        self.ffn_dim_expansion = ffn_dim_expansion
        self.num_transformer_layers = num_transformer_layers
        self.reg_refine = reg_refine
        self.num_reg_refine = num_reg_refine
        self.attn_type = attn_type
        self.attn_splits_list = attn_splits_list or [2]  # Simplified for speed
        self.corr_radius_list = corr_radius_list or [-1]  # Global correlation only
        self.prop_radius_list = prop_radius_list or [-1]  # Global propagation only
        
        # Apply PyTorch optimizations for RTX 3080
        self._apply_pytorch_optimizations()
        
        print(f"Initializing UniMatch Stereo depth extractor...")
        print(f"Device: {self.device}")
        print(f"Model: {self.model_checkpoint}")
        print(f"Batch size: {self.batch_size} (optimized for RTX 3080)")
        print(f"Scales: {self.num_scales}, Refinement: {self.num_reg_refine}")
        if self.inference_size:
            print(f"Inference size: {self.inference_size} (speed optimization)")
        
        # Show memory info
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"GPU Memory: {gpu_memory:.1f}GB")
        
        # Initialize model
        self.model = None
        self.model_loaded = False
        
        # Setup transforms
        self.val_transform_list = [
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ]
        self.val_transform = transforms.Compose(self.val_transform_list)
        
        # Memory management
        self.max_vram_usage = 0.9  # Use 90% of available VRAM
        self.memory_stats = defaultdict(float)
    
    def _apply_pytorch_optimizations(self):
        """ Apply RTX 3080 specific optimizations """
        if self.device == "cuda" and torch.cuda.is_available():
            # Enable tensor cores and optimizations
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cuda.matmul.allow_tf32 = True
            
            # Enable flash attention if available
            try:
                torch.backends.cuda.enable_flash_sdp(True)
            except:
                pass
            
            # Set memory fraction to avoid fragmentation
            torch.cuda.set_per_process_memory_fraction(0.95)
            
            print("✓ Applied RTX 3080 optimizations")
    
    def load_model(self):
        """ Load UniMatch stereo model to GPU """
        if self.model_loaded:
            return
        
        print(f"Loading UniMatch model from {self.model_checkpoint}...")
        
        # Initialize model
        self.model = UniMatch(
            feature_channels=self.feature_channels,
            num_scales=self.num_scales,
            upsample_factor=self.upsample_factor,
            num_head=self.num_head,
            ffn_dim_expansion=self.ffn_dim_expansion,
            num_transformer_layers=self.num_transformer_layers,
            reg_refine=self.reg_refine,
            task='stereo'
        ).to(self.device)
        
        # Load checkpoint
        checkpoint_path = Path(self.model_checkpoint)
        if not checkpoint_path.exists():
            raise ValueError(f"Checkpoint not found: {checkpoint_path}")
        
        try:
            checkpoint = torch.load(str(checkpoint_path), map_location=self.device)
            # Handle different checkpoint formats
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
            
            self.model.load_state_dict(state_dict, strict=False)
            print(f"✓ Loaded checkpoint: {checkpoint_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load checkpoint {checkpoint_path}: {e}")
        
        self.model.eval()
        self.model_loaded = True
        print("✓ UniMatch model loaded successfully")
    
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
    
    def preprocess_frame_pair(self, left_frame: np.ndarray, right_frame: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Preprocess frame pair for UniMatch """
        # Convert BGR to RGB
        left_rgb = cv2.cvtColor(left_frame, cv2.COLOR_BGR2RGB).astype(np.float32)
        right_rgb = cv2.cvtColor(right_frame, cv2.COLOR_BGR2RGB).astype(np.float32)
        
        # Apply transforms
        sample = {'left': left_rgb, 'right': right_rgb}
        sample = self.val_transform(sample)
        
        # Move to device and add batch dimension
        left_tensor = sample['left'].to(self.device).unsqueeze(0)  # [1, 3, H, W]
        right_tensor = sample['right'].to(self.device).unsqueeze(0)  # [1, 3, H, W]
        
        return left_tensor, right_tensor
    
    def preprocess_frame_batch(self, frame_pairs: List[Tuple[np.ndarray, np.ndarray]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Preprocess batch of frame pairs efficiently """
        left_batch = []
        right_batch = []
        
        for left_frame, right_frame in frame_pairs:
            # Convert BGR to RGB
            left_rgb = cv2.cvtColor(left_frame, cv2.COLOR_BGR2RGB).astype(np.float32)
            right_rgb = cv2.cvtColor(right_frame, cv2.COLOR_BGR2RGB).astype(np.float32)
            
            # Apply transforms
            sample = {'left': left_rgb, 'right': right_rgb}
            sample = self.val_transform(sample)
            
            left_batch.append(sample['left'])
            right_batch.append(sample['right'])
        
        # Stack into batch tensors
        left_tensor = torch.stack(left_batch, dim=0).to(self.device)  # [B, 3, H, W]
        right_tensor = torch.stack(right_batch, dim=0).to(self.device)  # [B, 3, H, W]
        
        return left_tensor, right_tensor
    
    def process_frame_batch(self, frame_pairs: List[Tuple[np.ndarray, np.ndarray]]) -> List[np.ndarray]:
        """ Process batch of frame pairs for depth estimation using UniMatch (OPTIMIZED) """
        if not self.model_loaded:
            self.load_model()
        
        if len(frame_pairs) == 0:
            return []
        
        depth_maps = []
        
        with torch.no_grad():
            try:
                # Process entire batch at once for massive speedup
                left_batch, right_batch = self.preprocess_frame_batch(frame_pairs)
                
                # Handle inference size and padding for entire batch
                ori_size = left_batch.shape[-2:]
                
                if self.inference_size is None:
                    # Use padding to make divisible by padding_factor
                    padder = InputPadder(left_batch.shape, padding_factor=self.padding_factor)
                    left_padded, right_padded = padder.pad(left_batch, right_batch)
                else:
                    # Resize to inference size for speed
                    left_padded = F.interpolate(left_batch, size=self.inference_size, 
                                              mode='bilinear', align_corners=True)
                    right_padded = F.interpolate(right_batch, size=self.inference_size, 
                                               mode='bilinear', align_corners=True)
                
                # Run UniMatch stereo on entire batch
                pred_disps = self.model(left_padded, right_padded,
                                      attn_type=self.attn_type,
                                      attn_splits_list=self.attn_splits_list,
                                      corr_radius_list=self.corr_radius_list,
                                      prop_radius_list=self.prop_radius_list,
                                      num_reg_refine=self.num_reg_refine,
                                      task='stereo')['flow_preds'][-1]  # [B, H, W]
                
                # Handle output size for entire batch
                if self.inference_size is None:
                    # Unpad result
                    pred_disps = padder.unpad(pred_disps)  # [B, H, W]
                else:
                    # Resize back to original size
                    pred_disps = F.interpolate(pred_disps.unsqueeze(1), size=ori_size, 
                                            mode='bilinear', align_corners=True).squeeze(1)  # [B, H, W]
                    # Scale disparity values
                    pred_disps = pred_disps * ori_size[-1] / float(self.inference_size[-1])
                
                # Convert batch to individual depth maps
                for i in range(pred_disps.shape[0]):
                    disp_np = pred_disps[i].cpu().numpy()
                    
                    # Convert disparity to depth
                    depth_map = np.where(disp_np > 0, 1.0 / (disp_np + 1e-6), 0)
                    
                    # Normalize depth to 0-1 range for saving
                    if depth_map.max() > depth_map.min():
                        depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
                    
                    depth_maps.append(depth_map.astype(np.float32))
                    
            except Exception as e:
                print(f"Error processing frame batch: {e}")
                # Create dummy depth maps on error
                for left_frame, right_frame in frame_pairs:
                    h, w = left_frame.shape[:2]
                    depth_maps.append(np.zeros((h, w), dtype=np.float32))
        
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
    parser = argparse.ArgumentParser(description='Extract depth maps from SBS stereoscopic video using UniMatch')
    parser.add_argument('video', help='Path to SBS video file')
    parser.add_argument('--start-frame', type=int, default=0,
                       help='Starting frame number (default: 0)')
    parser.add_argument('--max-frames', type=int, default=None,
                       help='Maximum number of frames to process (default: all)')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for GPU processing (default: 32 for RTX 3080)')
    parser.add_argument('--model', default="./models/gmstereo-scale2-regrefine3-resumeflowthings-middleburyfthighres-a82bec03.pth",
                       help='UniMatch model checkpoint path')
    parser.add_argument('--work-dir', default='temp_depth',
                       help='Working directory for output (default: temp_depth)')
    parser.add_argument('--force', action='store_true',
                       help='Force reprocessing even if cached results exist')
    parser.add_argument('--device', default='cuda',
                       help='Processing device (default: cuda)')
    parser.add_argument('--no-unsqueeze', action='store_true',
                       help='Skip SBS unsqueezing (keep squeezed aspect ratio)')
    parser.add_argument('--num-scales', type=int, default=1,
                       help='Number of feature scales (default: 1 for speed)')
    parser.add_argument('--num-reg-refine', type=int, default=1,
                       help='Number of refinement iterations (default: 1 for speed)')
    parser.add_argument('--inference-size', type=int, nargs=2, default=[480, 854],
                       help='Inference size [height width] (default: [480, 854] for speed)')
    
    args = parser.parse_args()
    
    # Handle flags
    unsqueeze_sbs = not args.no_unsqueeze
    
    # Optimization suggestions
    if args.batch_size > 16 and args.inference_size is None:
        print("⚠️  Large batch size with full resolution may cause OOM. Consider --inference-size 480 854")
    
    try:
        # Initialize depth extractor
        extractor = UniMatchStereoDepthExtractor(
            model_checkpoint=args.model,
            work_dir=args.work_dir,
            cache_dir=args.work_dir,
            device=args.device,
            batch_size=args.batch_size,
            unsqueeze_sbs=unsqueeze_sbs,
            num_scales=args.num_scales,
            num_reg_refine=args.num_reg_refine,
            inference_size=args.inference_size
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
