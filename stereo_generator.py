"""
Simple stereo generator with NVENC encoding and fixed quantile issue.
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
import time
import subprocess
import tempfile
from typing import Optional


def generate_stereo_video_nvenc(video_path: str, 
                               depth_path: str, 
                               output_path: str,
                               parallax_strength: float = 0.02,
                               chunk_size: int = 16,
                               start_frame: int = 0,
                               end_frame: Optional[int] = None,
                               max_frames: Optional[int] = None,
                               depth_start_frame: int = 0,  # NEW: depth video start frame
                               nvenc_preset: str = "p4",
                               crf: int = 23) -> str:
    """
    Generate Half-SBS stereo video with NVENC hardware encoding.
    
    Args:
        video_path: Path to input video
        depth_path: Path to depth map video  
        output_path: Path for output stereo video
        parallax_strength: 3D effect strength (0.01-0.05)
        chunk_size: Frames to process at once
        start_frame: Starting frame number in source video (0-based)
        end_frame: Ending frame number in source video (exclusive, None for end)
        max_frames: Max frames to process (overrides end_frame if smaller)
        depth_start_frame: Starting frame in depth video (usually 0)
        nvenc_preset: NVENC preset p1-p7 (p4 balanced)
        crf: Quality 18-28 (lower=better)
    """
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Clear GPU memory from previous operations
    if device.type == "cuda":
        torch.cuda.empty_cache()
        print(f"🧹 Cleared CUDA cache")
        print(f"GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f}GB allocated, "
              f"{torch.cuda.memory_reserved() / 1e9:.2f}GB reserved")
    
    print(f"🚀 Stereo generation with NVENC on {device}")
    print(f"Input: {video_path}")
    print(f"Depth: {depth_path}")
    print(f"Output: {output_path}")
    
    # Open video readers
    video_cap = cv2.VideoCapture(str(video_path))
    depth_cap = cv2.VideoCapture(str(depth_path))
    
    if not video_cap.isOpened() or not depth_cap.isOpened():
        raise RuntimeError("Failed to open input videos")
    
    # Get video properties
    fps = video_cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Calculate frame range
    if end_frame is None:
        end_frame = total_frames
    else:
        end_frame = min(end_frame, total_frames)
    
    # Ensure valid range
    start_frame = max(0, start_frame)
    end_frame = max(start_frame, end_frame)
    
    frame_count = end_frame - start_frame
    
    # Apply max_frames limit if specified
    if max_frames is not None:
        frame_count = min(frame_count, max_frames)
        end_frame = start_frame + frame_count
    
    print(f"Frame range: {start_frame} to {end_frame-1} ({frame_count} frames)")
    print(f"Processing {frame_count} frames at {width}x{height}, {fps} FPS")
    
    # Seek to start frames
    print(f"Seeking video to frame {start_frame}, depth to frame {depth_start_frame}")
    video_cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    depth_cap.set(cv2.CAP_PROP_POS_FRAMES, depth_start_frame)
    
    # Create temporary raw video file
    temp_dir = tempfile.mkdtemp()
    temp_raw = Path(temp_dir) / "stereo_raw.yuv"
    
    start_time = time.time()
    frames_processed = 0
    
    try:
        # Open raw file for writing
        with open(temp_raw, 'wb') as raw_file:
            
            while frames_processed < frame_count:
                # Read chunk of frames
                video_frames = []
                depth_frames = []
                
                actual_chunk_size = min(chunk_size, frame_count - frames_processed)
                
                for i in range(actual_chunk_size):
                    ret_vid, frame = video_cap.read()
                    ret_dep, depth = depth_cap.read()
                    
                    # Debug: check if reads are successful
                    if i == 0:  # Only print for first frame in chunk
                        print(f"\nReading frame {start_frame + frames_processed + i}: video={ret_vid}, depth={ret_dep}")
                        #if ret_vid:
                        #    print(f"Video frame shape: {frame.shape}")
                        #if ret_dep:
                        #    print(f"Depth frame shape: {depth.shape}")
                    
                    if not ret_vid or not ret_dep:
                        print(f"\nFrame read failed at frame {start_frame + frames_processed + i}")
                        print(f"Video ret: {ret_vid}, Depth ret: {ret_dep}")
                        break
                    
                    # Convert to RGB and normalize
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                    depth_gray = cv2.cvtColor(depth, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
                    
                    video_frames.append(frame_rgb)
                    depth_frames.append(depth_gray)
                
                print(f"Read {len(video_frames)} frames in this chunk")
                
                if not video_frames:
                    print("No frames read, breaking")
                    break
                
                # Convert to tensors
                frames_tensor = torch.from_numpy(np.stack(video_frames)).permute(0, 3, 1, 2).to(device)
                depths_tensor = torch.from_numpy(np.stack(depth_frames)).unsqueeze(1).to(device)
                
                # Generate stereo frames
                stereo_frames = create_stereo_chunk(frames_tensor, depths_tensor, parallax_strength)
                
                # Convert to YUV420 and write to raw file
                write_yuv420_chunk(stereo_frames, raw_file)
                
                frames_processed += len(video_frames)
                
                # Progress update
                current_frame = start_frame + frames_processed
                elapsed = time.time() - start_time
                fps_current = frames_processed / elapsed if elapsed > 0 else 0
                eta = (frame_count - frames_processed) / fps_current if fps_current > 0 else 0
                
                print(f"\rProgress: {frames_processed}/{frame_count} "
                      f"(frame {current_frame}/{end_frame-1}) "
                      f"{100*frames_processed/frame_count:.1f}% "
                      f"Speed: {fps_current:.1f} FPS "
                      f"ETA: {eta:.0f}s", end='', flush=True)
        
        print(f"\n🎬 Encoding with NVENC...")
        
        # Encode with NVENC
        encode_with_nvenc(temp_raw, output_path, width, height, fps, nvenc_preset, crf)
        
    finally:
        video_cap.release()
        depth_cap.release()
        
        # Cleanup temp files
        if temp_raw.exists():
            temp_raw.unlink()
        Path(temp_dir).rmdir()
    
    total_time = time.time() - start_time
    print(f"✅ NVENC stereo generation complete in {total_time:.1f}s")
    print(f"Average speed: {frames_processed/total_time:.1f} FPS")
    
    return str(output_path)


def create_stereo_chunk(frames: torch.Tensor, depths: torch.Tensor, parallax_strength: float) -> torch.Tensor:
    """Convert frames to Half-SBS stereo on GPU."""
    
    B, C, H, W = frames.shape
    device = frames.device
    
    # Fixed depth normalization (no quantile issues)
    depths_norm = normalize_depth_fixed(depths)
    
    # Create coordinate grids
    y_coords, x_coords = torch.meshgrid(
        torch.linspace(-1, 1, H, device=device),
        torch.linspace(-1, 1, W, device=device),
        indexing='ij'
    )
    
    # Create horizontal shift based on depth
    max_shift_pixels = parallax_strength * W
    shift_amounts = depths_norm.squeeze(1) * max_shift_pixels
    shift_normalized = (shift_amounts * 2.0) / W
    
    # Create sampling grid for right eye
    x_shifted = x_coords.unsqueeze(0).repeat(B, 1, 1) + shift_normalized
    x_shifted = torch.clamp(x_shifted, -1, 1)
    
    y_grid = y_coords.unsqueeze(0).repeat(B, 1, 1)
    grid_right = torch.stack([x_shifted, y_grid], dim=-1)
    
    # Generate right eye view
    right_eye = F.grid_sample(frames, grid_right, mode='bilinear', 
                             padding_mode='border', align_corners=True)
    
    # Create Half-SBS output
    left_half = F.interpolate(frames, size=(H, W//2), mode='bilinear', align_corners=False)
    right_half = F.interpolate(right_eye, size=(H, W//2), mode='bilinear', align_corners=False)
    
    return torch.cat([left_half, right_half], dim=3)


def normalize_depth_fixed(depth: torch.Tensor) -> torch.Tensor:
    """Fixed depth normalization without quantile memory issues."""
    
    # Flatten and sample for percentile calculation
    depth_flat = depth.flatten()
    
    if depth_flat.numel() > 100000:
        # Sample random subset
        sample_size = min(10000, depth_flat.numel())
        indices = torch.randperm(depth_flat.numel(), device=depth.device)[:sample_size]
        sample = depth_flat[indices]
        
        # Manual percentile calculation
        sample_sorted = torch.sort(sample)[0]
        p2_idx = max(0, int(0.02 * len(sample_sorted)))
        p98_idx = min(len(sample_sorted)-1, int(0.98 * len(sample_sorted)))
        
        depth_min = sample_sorted[p2_idx]
        depth_max = sample_sorted[p98_idx]
    else:
        # Small tensor, use direct min/max
        depth_min = depth.min()
        depth_max = depth.max()
    
    # Normalize
    if depth_max > depth_min:
        return (torch.clamp(depth, depth_min, depth_max) - depth_min) / (depth_max - depth_min)
    else:
        return torch.zeros_like(depth)


def write_yuv420_chunk(stereo_frames: torch.Tensor, raw_file):
    """Convert RGB frames to YUV420 and write to file."""
    
    # Convert to CPU and denormalize
    frames_cpu = (stereo_frames.clamp(0, 1) * 255).cpu().byte()
    B, C, H, W = frames_cpu.shape
    
    for i in range(B):
        # Get RGB frame
        rgb_frame = frames_cpu[i].permute(1, 2, 0).numpy()  # [H, W, 3]
        
        # Convert RGB to YUV420
        yuv_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2YUV_I420)
        
        # Write to file
        raw_file.write(yuv_frame.tobytes())


def encode_with_nvenc(raw_file: Path, output_path: str, width: int, height: int, 
                     fps: float, preset: str, crf: int):
    """Encode raw YUV420 file using NVENC."""
    
    cmd = [
        'ffmpeg', '-y',
        '-f', 'rawvideo',
        '-pix_fmt', 'yuv420p',
        '-s', f'{width}x{height}',
        '-r', str(fps),
        '-i', str(raw_file),
        '-c:v', 'hevc_nvenc',
        '-preset', 'slow',
        '-crf', '25',
        '-movflags', '+faststart',
        output_path
    ]
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"FFmpeg error: {result.stderr}")
        raise RuntimeError(f"NVENC encoding failed: {result.stderr}")


if __name__ == "__main__":
    generate_stereo_video_nvenc(
        video_path="./everest/Everest-2160p-2D.mp4",
        depth_path="./everest/depth_4k_final.mp4", 
        output_path="output_3d_half_sbs_nvenc.mp4",
        parallax_strength=0.025,
        chunk_size=4,         # Reduced for 4K frames
        start_frame=8400,     # Start at frame 8400 in source video
        end_frame=8900,       # End at frame 8900 in source video
        depth_start_frame=0,  # Depth video starts at frame 0
        nvenc_preset="p4",
        crf=20
    )