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
import argparse
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
                
                #print(f"Read {len(video_frames)} frames in this chunk")
                
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
    """
    Create Half-SBS stereo using single-eye approach:
    - Left eye: Original frame (perfect quality)
    - Right eye: Depth-warped frame
    """
    
    B, C, H, W = frames.shape
    device = frames.device
    
    # Normalize depth maps with robust percentile method
    depths_norm = normalize_depth_robust(depths)
    
    # Estimate subject depth for zero-parallax adjustment
    subject_depths = estimate_subject_depth_batch(depths_norm)
    
    # Generate right eye by warping original frames
    right_eyes = []
    
    for i in range(B):
        frame = frames[i]  # [C, H, W]
        depth = depths_norm[i]  # [1, H, W]
        subject_depth = subject_depths[i]  # scalar
        
        # Create right eye using forward warping
        right_eye = forward_warp_right_eye(frame, depth, subject_depth, parallax_strength, W)
        right_eyes.append(right_eye)
    
    right_eyes_tensor = torch.stack(right_eyes)
    
    # Create Half-SBS output
    # Left half: original frames (downscaled) - PERFECT quality
    left_half = F.interpolate(frames, size=(H, W//2), mode='bilinear', align_corners=False)
    
    # Right half: warped frames (downscaled)
    right_half = F.interpolate(right_eyes_tensor, size=(H, W//2), mode='bilinear', align_corners=False)
    
    # Concatenate horizontally for Half-SBS
    return torch.cat([left_half, right_half], dim=3)


def normalize_depth_robust(depth: torch.Tensor) -> torch.Tensor:
    """
    Robust depth normalization using percentiles to handle outliers.
    """
    
    B = depth.shape[0]
    normalized = torch.zeros_like(depth)
    
    for i in range(B):
        d = depth[i].flatten()
        
        # Use percentiles for robust normalization
        if d.numel() > 1000:
            # Sample for efficiency on large tensors
            sample_size = min(5000, d.numel())
            indices = torch.randperm(d.numel(), device=d.device)[:sample_size]
            sample = d[indices]
            sample_sorted = torch.sort(sample)[0]
            
            p5_idx = max(0, int(0.05 * len(sample_sorted)))
            p95_idx = min(len(sample_sorted)-1, int(0.95 * len(sample_sorted)))
            
            d_min = sample_sorted[p5_idx]
            d_max = sample_sorted[p95_idx]
        else:
            d_min = d.quantile(0.05)
            d_max = d.quantile(0.95)
        
        # Normalize to [0, 1]
        if d_max > d_min:
            normalized[i] = (torch.clamp(depth[i], d_min, d_max) - d_min) / (d_max - d_min)
        else:
            normalized[i] = torch.ones_like(depth[i]) * 0.5
    
    return normalized


def forward_warp_right_eye(frame: torch.Tensor, depth: torch.Tensor, 
                          subject_depth: float, parallax_strength: float, width: int) -> torch.Tensor:
    """
    Forward warp frame to create right eye view using depth-based disparity.
    """
    
    C, H, W = frame.shape
    device = frame.device
    
    # Calculate disparity based on depth difference from subject
    depth_diff = depth.squeeze(0) - subject_depth  # [H, W]
    
    # Apply smoothing to reduce aliasing
    depth_diff = F.avg_pool2d(
        depth_diff.unsqueeze(0).unsqueeze(0), 
        kernel_size=3, stride=1, padding=1
    ).squeeze()
    
    # Convert depth difference to pixel disparity with reduced strength
    max_disparity_pixels = parallax_strength * width * 0.5  # Reduce by half
    disparity = depth_diff * max_disparity_pixels  # [H, W]
    
    # Quantize disparity to reduce sub-pixel oscillations
    disparity = torch.round(disparity * 4.0) / 4.0  # Quarter-pixel precision
    
    # Create sampling grid for right eye
    y_coords, x_coords = torch.meshgrid(
        torch.linspace(-1, 1, H, device=device),
        torch.linspace(-1, 1, W, device=device), 
        indexing='ij'
    )
    
    # Convert pixel disparity to normalized coordinates
    disparity_norm = (disparity * 2.0) / W  # Convert to [-1, 1] range
    
    # Apply disparity shift with clamping
    x_shifted = x_coords - disparity_norm  # Subtract for right eye view
    x_shifted = torch.clamp(x_shifted, -0.99, 0.99)  # Tighter clamp to avoid edge artifacts
    
    # Create sampling grid
    grid = torch.stack([x_shifted, y_coords], dim=-1)  # [H, W, 2]
    grid = grid.unsqueeze(0)  # [1, H, W, 2]
    frame_batch = frame.unsqueeze(0)  # [1, C, H, W]
    
    # Sample right eye view with antialiasing
    right_eye = F.grid_sample(frame_batch, grid, mode='bilinear', 
                             padding_mode='reflection', align_corners=False)  # Use reflection padding
    
    right_eye = right_eye.squeeze(0)  # [C, H, W]
    
    # Apply gentle smoothing to reduce remaining artifacts
    right_eye = F.avg_pool2d(
        right_eye.unsqueeze(0), 
        kernel_size=3, stride=1, padding=1
    ).squeeze(0) * 0.7 + right_eye * 0.3
    
    # Apply hole filling
    right_eye = fill_holes_simple(right_eye, frame)
    
    return right_eye


def estimate_subject_depth_batch(depths: torch.Tensor) -> torch.Tensor:
    """
    Estimate subject depth for each frame in batch using center-weighted approach.
    """
    
    B, _, H, W = depths.shape
    subject_depths = torch.zeros(B, device=depths.device)
    
    for i in range(B):
        depth = depths[i, 0]  # [H, W]
        
        # Focus on center region (typical subject location)
        center_h_start, center_h_end = H//4, H*3//4
        center_w_start, center_w_end = W//4, W*3//4
        
        center_region = depth[center_h_start:center_h_end, center_w_start:center_w_end]
        
        # Use histogram peak to find dominant depth (likely subject)
        valid_depths = center_region[(center_region > 0.1) & (center_region < 0.9)]
        
        if valid_depths.numel() > 20:
            # Find histogram peak
            hist = torch.histc(valid_depths, bins=32, min=0.0, max=1.0)
            peak_bin = torch.argmax(hist)
            subject_depth = (peak_bin.float() + 0.5) / 32.0
            
            # Blend with median for stability
            median_depth = torch.median(valid_depths)
            subject_depths[i] = 0.7 * subject_depth + 0.3 * median_depth
        else:
            subject_depths[i] = 0.5  # fallback
    
    return subject_depths


def fill_holes_simple(warped_frame: torch.Tensor, original_frame: torch.Tensor) -> torch.Tensor:
    """
    Simple hole filling by blending with original frame where artifacts are detected.
    """
    
    C, H, W = warped_frame.shape
    
    # Detect potential holes/artifacts by looking for high-frequency noise
    gray_warped = warped_frame.mean(dim=0)  # [H, W]
    
    # Compute gradient magnitude to detect discontinuities
    grad_x = F.pad(gray_warped[:, 1:] - gray_warped[:, :-1], (1, 0))
    grad_y = F.pad(gray_warped[1:, :] - gray_warped[:-1, :], (0, 0, 1, 0))
    grad_mag = torch.sqrt(grad_x**2 + grad_y**2)
    
    # Create artifact mask (high gradient areas likely have artifacts)
    artifact_threshold = 0.1
    artifact_mask = (grad_mag > artifact_threshold).float()
    
    # Smooth the mask to avoid hard edges
    kernel_size = 5
    artifact_mask = F.avg_pool2d(
        artifact_mask.unsqueeze(0).unsqueeze(0), 
        kernel_size, stride=1, padding=kernel_size//2
    ).squeeze()
    
    # Expand mask to all channels
    artifact_mask = artifact_mask.unsqueeze(0).expand(C, -1, -1)
    
    # Blend warped frame with original where artifacts detected
    blend_strength = 0.3
    filled_frame = (1 - blend_strength * artifact_mask) * warped_frame + \
                   (blend_strength * artifact_mask) * original_frame
    
    return filled_frame


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
        '-crf', '20',
        '-movflags', '+faststart',
        output_path
    ]
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"FFmpeg error: {result.stderr}")
        raise RuntimeError(f"NVENC encoding failed: {result.stderr}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Optimized 3D video pipeline')
    parser.add_argument('video_path', help='Path to 4K 2D video')
    parser.add_argument('depth_path', help='Path to 4K depth video')
    parser.add_argument('--start-frame', type=int, default=0,
                       help='Starting frame number (default: 0)')
    parser.add_argument('--max-frames', type=int,
                       help='Maximum frames to process')
    parser.add_argument('--output-path', default="./video_3D_4K.mp4",
                       help='Output path (default: video_3D_4K.mp4')
    
    args = parser.parse_args()
    if args.max_frames:
        end_frame = args.start_frame + args.max_frames
    else:
        end_frame = None
    
    try:
        generate_stereo_video_nvenc(
            video_path=args.video_path,
            depth_path=args.depth_path, 
            output_path=args.output_path,
            parallax_strength=0.01,
            chunk_size=12,
            start_frame=args.start_frame,
            end_frame=end_frame,
            depth_start_frame=args.start_frame,
            nvenc_preset="p4",
            crf=23
            )        
        print("\n🎉 Stereo video completed successfully!")
        
    except Exception as e:
        print(f"\n💥 Video creation failed: {e}")
        import traceback
        traceback.print_exc()

