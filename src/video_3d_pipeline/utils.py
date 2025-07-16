"""Shared utilities for video processing."""

import cv2
import ffmpeg
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import tempfile
import os
import hashlib
import librosa
from scipy import signal
import soundfile as sf
import matplotlib.pyplot as plt


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


def extract_audio(video_path: str, work_dir: Path, 
                 duration_seconds: float = 600, sample_rate: int = 22050) -> str:
    """ Extract audio from video for sync analysis with caching """
    
    # Create cache filename based on video path and parameters
    video_hash = hashlib.md5(f"{video_path}_{duration_seconds}_{sample_rate}".encode()).hexdigest()[:16]
    audio_cache_path = work_dir / f"audio_cache_{video_hash}.wav"
    
    # Check if cached audio exists and is newer than video
    if audio_cache_path.exists():
        video_mtime = os.path.getmtime(video_path)
        audio_mtime = os.path.getmtime(audio_cache_path)
        if audio_mtime > video_mtime:
            print(f"Using cached audio: {audio_cache_path}")
            return str(audio_cache_path)
    
    print(f"Extracting audio from {video_path}...")
    
    # Extract audio using FFmpeg
    # Only extract first portion to save time/space
    (
        ffmpeg
        .input(video_path, t=duration_seconds)
        .output(str(audio_cache_path), 
               acodec='pcm_s16le', 
               ar=sample_rate, 
               ac=1,  # mono
               y='-')  # overwrite
        .run(quiet=True)
    )
    
    if not audio_cache_path.exists():
        raise ValueError(f"Failed to extract audio from {video_path}")
        
    return str(audio_cache_path)


def load_audio_for_sync(audio_path: str, max_length_seconds: float = 300) -> Tuple[np.ndarray, int]:
    """ Load audio waveform for synchronization analysis """
    
    # Load audio with librosa
    audio, sr = librosa.load(audio_path, sr=None, mono=True)
    
    # Limit to max_length to keep processing fast
    max_samples = int(max_length_seconds * sr)
    if len(audio) > max_samples:
        audio = audio[:max_samples]
        print(f"Limited audio to {max_length_seconds}s for sync analysis")
    
    return audio, sr


def find_audio_offset(audio1: np.ndarray, audio2: np.ndarray, sr: int) -> Tuple[float, float]:
    """ Find time offset between two audio tracks using cross-correlation """
    
    print("Computing audio cross-correlation...")
    
    # Normalize audio to prevent overflow
    audio1_norm = audio1 / (np.max(np.abs(audio1)) + 1e-10)
    audio2_norm = audio2 / (np.max(np.abs(audio2)) + 1e-10)
    
    # Cross-correlate the signals
    correlation = signal.correlate(audio2_norm, audio1_norm, mode='full')
    
    # Find peak correlation
    max_corr_idx = np.argmax(np.abs(correlation))
    max_corr_value = correlation[max_corr_idx]
    
    # Convert sample offset to time offset
    sample_offset = max_corr_idx - len(audio1) + 1
    time_offset = sample_offset / sr
    
    # Correlation strength (normalized)
    correlation_strength = float(np.abs(max_corr_value)) / len(audio1)
    
    print(f"Audio offset: {time_offset:.3f}s, correlation strength: {correlation_strength:.4f}")
    
    return time_offset, correlation_strength


def plot_audio_correlation(audio1: np.ndarray, audio2: np.ndarray, sr: int, 
                          time_offset: float, work_dir: Path):
    """ Plot audio waveforms and correlation for visualization """
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 8))
    
    # Plot audio waveforms
    time1 = np.arange(len(audio1)) / sr
    time2 = np.arange(len(audio2)) / sr
    
    ax1.plot(time1, audio1, alpha=0.7, label='Video 1')
    ax1.set_ylabel('Amplitude')
    ax1.set_title('Audio Waveform - Video 1')
    ax1.grid(True)
    
    ax2.plot(time2, audio2, alpha=0.7, label='Video 2', color='orange')
    ax2.set_ylabel('Amplitude')
    ax2.set_title('Audio Waveform - Video 2')
    ax2.grid(True)
    
    # Plot correlation
    correlation = signal.correlate(audio2, audio1, mode='full')
    corr_time = (np.arange(len(correlation)) - len(audio1) + 1) / sr
    
    ax3.plot(corr_time, correlation)
    ax3.axvline(time_offset, color='red', linestyle='--', 
               label=f'Best offset: {time_offset:.3f}s')
    ax3.set_xlabel('Time Offset (seconds)')
    ax3.set_ylabel('Correlation')
    ax3.set_title('Audio Cross-Correlation')
    ax3.legend()
    ax3.grid(True)
    
    plt.tight_layout()
    plt.savefig(work_dir / 'audio_sync_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()


def verify_video_compatibility(video1_path: str, video2_path: str) -> bool:
    """ Check if videos are compatible for sync (duration, frame rate, etc.) """
    
    info1 = get_video_info(video1_path)
    info2 = get_video_info(video2_path)
    
    if not info1 or not info2:
        print("Error: Could not read video information")
        return False
    
    # Check durations (should be within 2% of each other)
    duration_diff = abs(info1['duration'] - info2['duration'])
    duration_ratio = duration_diff / max(info1['duration'], info2['duration'])
    
    if duration_ratio > 0.02:  # More than 2% difference
        print(f"Warning: Large duration difference: {info1['duration']:.1f}s vs {info2['duration']:.1f}s")
        print("Videos may not be from the same source")
        return False
    
    # Check frame rates (should be identical or very close)
    fps_diff = abs(info1['fps'] - info2['fps'])
    if fps_diff > 0.1:
        print(f"Warning: Frame rate mismatch: {info1['fps']:.2f} vs {info2['fps']:.2f}")
        print("Consider re-encoding one video to match frame rates")
        return False
    
    print("✓ Videos appear compatible for synchronization")
    print(f"  Duration: {info1['duration']:.1f}s vs {info2['duration']:.1f}s")
    print(f"  Frame rate: {info1['fps']:.2f} vs {info2['fps']:.2f}")
    print(f"  Resolution: {info1['width']}x{info1['height']} vs {info2['width']}x{info2['height']}")
    
    return True


def create_work_directory(base_path: str = "temp_pipeline") -> Path:
    """ Create and return working directory path """
    work_dir = Path(base_path)
    work_dir.mkdir(exist_ok=True)
    return work_dir