"""Shared utilities for video processing."""

import cv2
import ffmpeg
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import tempfile
import os

try:
    import cupy as cp
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False


def get_video_info(video_path: str) -> Optional[Dict]:
    """ Get basic video information using ffprobe """
    try:
        probe = ffmpeg.probe(video_path)
        video_stream = next(
            (stream for stream in probe['streams'] if stream['codec_type'] == 'video'), 
            None
        )
        
        if not video_stream:
            return None
            
        return {
            'width': int(video_stream['width']),
            'height': int(video_stream['height']),
            'fps': eval(video_stream['r_frame_rate']),
            'duration': float(video_stream['duration']),
            'frames': int(video_stream.get('nb_frames', 0))
        }
    except Exception as e:
        print(f"Error getting video info: {e}")
        return None


def extract_frames(video_path: str, interval_seconds: float = 1.0, 
                  resize_height: int = 480) -> Tuple[List, List]:
    """ Extract frames efficiently using GPU acceleration and batched processing """
    
    # Get video info
    video_info = get_video_info(video_path)
    if not video_info:
        raise ValueError(f"Could not get video info for: {video_path}")
    
    duration = video_info['duration']
    width = video_info['width']
    height = video_info['height']
    
    # Calculate target dimensions maintaining aspect ratio
    new_width = int(width * resize_height / height)
    
    # Generate timestamps - but be more conservative to avoid overheating
    max_frames = min(300, int(duration / interval_seconds))  # Limit to 300 frames max
    if max_frames < 10:
        max_frames = min(60, int(duration))  # At least try for 60 frames
        
    target_times = np.linspace(10, duration - 10, max_frames)  # Avoid start/end
    
    print(f"Extracting {len(target_times)} keyframes from {duration:.1f}s video (GPU accelerated)...")
    
    frames = []
    timestamps = []
    
    # Check if we can use CUDA acceleration
    use_cuda = CUDA_AVAILABLE and cv2.cuda.getCudaEnabledDeviceCount() > 0
    if use_cuda:
        print("Using CUDA GPU acceleration")
    
    # Process in batches to avoid thermal issues
    batch_size = 20  # Process 20 frames at a time
    for batch_start in range(0, len(target_times), batch_size):
        batch_end = min(batch_start + batch_size, len(target_times))
        batch_times = target_times[batch_start:batch_end]
        
        print(f"Processing batch {batch_start//batch_size + 1}/{(len(target_times)-1)//batch_size + 1}")
        
        # Extract this batch using a single FFmpeg call
        batch_frames, batch_timestamps = _extract_frame_batch(
            video_path, batch_times, new_width, resize_height, use_cuda
        )
        
        frames.extend(batch_frames)
        timestamps.extend(batch_timestamps)
    
    if not frames:
        raise ValueError("No frames could be extracted from video")
        
    print(f"Successfully extracted {len(frames)} frames")
    return frames, timestamps


def _extract_frame_batch(video_path: str, timestamps: np.ndarray, 
                        new_width: int, resize_height: int, use_cuda: bool) -> Tuple[List, List]:
    """ Extract a batch of frames using GPU acceleration """
    frames = []
    batch_timestamps = []
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Extract all frames in this batch to temporary files
        for i, timestamp in enumerate(timestamps):
            output_file = os.path.join(temp_dir, f"frame_{i:04d}.jpg")
            
            try:
                # Use CUDA decoder if available
                input_args = {}
                if use_cuda:
                    input_args['hwaccel'] = 'cuda'
                    input_args['hwaccel_output_format'] = 'cuda'
                
                (
                    ffmpeg
                    .input(video_path, ss=timestamp, **input_args)
                    .output(output_file, vframes=1, qscale=2, s=f'{new_width}x{resize_height}')
                    .run(quiet=True, overwrite_output=True)
                )
                
                # Read and process the frame
                if os.path.exists(output_file):
                    if use_cuda:
                        frame = _read_frame_gpu(output_file)
                    else:
                        frame = _read_frame_cpu(output_file)
                    
                    if frame is not None:
                        frames.append(frame)
                        batch_timestamps.append(timestamp)
                        
            except Exception as e:
                print(f"Warning: Failed to extract frame at {timestamp:.1f}s: {e}")
                continue
    
    return frames, batch_timestamps


def _read_frame_gpu(image_path: str) -> Optional[np.ndarray]:
    """ Read and process frame using GPU """
    try:
        # Read image on CPU first
        img_cpu = cv2.imread(image_path)
        if img_cpu is None:
            return None
            
        # Upload to GPU
        img_gpu = cv2.cuda_GpuMat()
        img_gpu.upload(img_cpu)
        
        # Convert to grayscale on GPU
        gray_gpu = cv2.cuda.cvtColor(img_gpu, cv2.COLOR_BGR2GRAY)
        
        # Download result back to CPU
        gray_cpu = gray_gpu.download()
        
        return gray_cpu
        
    except Exception as e:
        print(f"GPU processing failed, falling back to CPU: {e}")
        return _read_frame_cpu(image_path)


def _read_frame_cpu(image_path: str) -> Optional[np.ndarray]:
    """ Read and process frame using CPU """
    try:
        img = cv2.imread(image_path)
        if img is None:
            return None
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    except:
        return None


def create_work_directory(base_path: str = "temp_pipeline") -> Path:
    """ Create and return working directory path """
    work_dir = Path(base_path)
    work_dir.mkdir(exist_ok=True)
    return work_dir


def compute_frame_correlation(frame1, frame2) -> float:
    """ Compute normalized cross-correlation between two frames """
    # Resize frames to same size
    h1, w1 = frame1.shape
    h2, w2 = frame2.shape
    target_h, target_w = min(h1, h2), min(w1, w2)
    
    frame1_resized = cv2.resize(frame1, (target_w, target_h))
    frame2_resized = cv2.resize(frame2, (target_w, target_h))
    
    # Compute normalized cross-correlation
    return cv2.matchTemplate(frame1_resized, frame2_resized, cv2.TM_CCOEFF_NORMED)[0][0]