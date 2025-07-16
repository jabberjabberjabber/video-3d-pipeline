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
    
    # Check if video has audio first
    video_info = get_video_info(video_path)
    if not video_info:
        raise ValueError(f"Could not read video info for {video_path}")
    
    # Check for audio streams
    try:
        probe = ffmpeg.probe(video_path)
        audio_streams = [stream for stream in probe['streams'] if stream['codec_type'] == 'audio']
        if not audio_streams:
            raise ValueError(f"No audio stream found in {video_path}")
        print(f"Found {len(audio_streams)} audio stream(s)")
    except Exception as e:
        raise ValueError(f"Error checking audio streams: {e}")
    
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
    
    try:
        # Extract audio using FFmpeg with better error handling
        (
            ffmpeg
            .input(video_path, t=duration_seconds)
            .output(str(audio_cache_path), 
                   acodec='pcm_s16le', 
                   ar=sample_rate, 
                   ac=1)  # mono
            .overwrite_output()
            .run(quiet=False, capture_stdout=True, capture_stderr=True)
        )
        
    except ffmpeg.Error as e:
        print(f"FFmpeg error details:")
        print(f"stdout: {e.stdout.decode() if e.stdout else 'None'}")
        print(f"stderr: {e.stderr.decode() if e.stderr else 'None'}")
        
        # Try alternative extraction method
        print("Trying alternative audio extraction...")
        try:
            (
                ffmpeg
                .input(video_path)
                .output(str(audio_cache_path), 
                       acodec='pcm_s16le', 
                       ar=sample_rate, 
                       ac=1,
                       t=duration_seconds)
                .overwrite_output()
                .run(quiet=False, capture_stdout=True, capture_stderr=True)
            )
        except ffmpeg.Error as e2:
            print(f"Alternative method also failed:")
            print(f"stderr: {e2.stderr.decode() if e2.stderr else 'None'}")
            raise ValueError(f"Could not extract audio from {video_path}")
    
    if not audio_cache_path.exists():
        raise ValueError(f"Audio extraction failed - output file not created")
        
    # Check if the output file has reasonable size
    if audio_cache_path.stat().st_size < 1000:  # Less than 1KB
        raise ValueError(f"Audio extraction produced unusually small file")
        
    print(f"Audio extracted successfully: {audio_cache_path}")
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
    
    # Normalize audio to zero mean and unit variance for better correlation
    audio1_norm = (audio1 - np.mean(audio1)) / (np.std(audio1) + 1e-10)
    audio2_norm = (audio2 - np.mean(audio2)) / (np.std(audio2) + 1e-10)
    
    # Cross-correlate the signals
    correlation = signal.correlate(audio2_norm, audio1_norm, mode='full', method='auto')
    
    # Find peak correlation
    max_corr_idx = np.argmax(np.abs(correlation))
    max_corr_value = correlation[max_corr_idx]
    
    # Convert sample offset to time offset
    sample_offset = max_corr_idx - len(audio1) + 1
    time_offset = sample_offset / sr
    
    # Proper correlation coefficient calculation
    # Normalize by the autocorrelation at zero lag
    auto_corr1 = np.sum(audio1_norm * audio1_norm)
    auto_corr2 = np.sum(audio2_norm * audio2_norm)
    correlation_strength = float(np.abs(max_corr_value)) / np.sqrt(auto_corr1 * auto_corr2)
    
    print(f"Audio offset: {time_offset:.3f}s, correlation strength: {correlation_strength:.4f}")
    
    return time_offset, correlation_strength


def plot_audio_correlation(audio1: np.ndarray, audio2: np.ndarray, sr: int, 
                          time_offset: float, work_dir: Path):
    """ Plot audio waveforms and correlation for visualization """
    
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 10))
    
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
    
    # Normalize audio for correlation plot
    audio1_norm = (audio1 - np.mean(audio1)) / (np.std(audio1) + 1e-10)
    audio2_norm = (audio2 - np.mean(audio2)) / (np.std(audio2) + 1e-10)
    
    # Plot correlation
    correlation = signal.correlate(audio2_norm, audio1_norm, mode='full')
    corr_time = (np.arange(len(correlation)) - len(audio1) + 1) / sr
    
    ax3.plot(corr_time, correlation)
    ax3.axvline(time_offset, color='red', linestyle='--', 
               label=f'Best offset: {time_offset:.3f}s')
    ax3.set_xlabel('Time Offset (seconds)')
    ax3.set_ylabel('Correlation')
    ax3.set_title('Audio Cross-Correlation (Normalized)')
    ax3.legend()
    ax3.grid(True)
    
    # Plot zoomed-in correlation around the peak
    peak_idx = np.argmax(np.abs(correlation))
    zoom_range = min(sr * 10, len(correlation) // 4)  # ±10 seconds or 1/4 of total
    start_idx = max(0, peak_idx - zoom_range)
    end_idx = min(len(correlation), peak_idx + zoom_range)
    
    zoom_corr = correlation[start_idx:end_idx]
    zoom_time = corr_time[start_idx:end_idx]
    
    ax4.plot(zoom_time, zoom_corr)
    ax4.axvline(time_offset, color='red', linestyle='--', 
               label=f'Best offset: {time_offset:.3f}s')
    ax4.set_xlabel('Time Offset (seconds)')
    ax4.set_ylabel('Correlation')
    ax4.set_title('Audio Cross-Correlation (Zoomed)')
    ax4.legend()
    ax4.grid(True)
    
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


def calculate_audio_correlation(audio1: np.ndarray, audio2: np.ndarray) -> float:
    """ Calculate normalized correlation coefficient between two audio signals """
    
    # Ensure both arrays are the same length
    min_len = min(len(audio1), len(audio2))
    audio1_trim = audio1[:min_len]
    audio2_trim = audio2[:min_len]
    
    # Normalize to zero mean and unit variance
    audio1_norm = (audio1_trim - np.mean(audio1_trim)) / (np.std(audio1_trim) + 1e-10)
    audio2_norm = (audio2_trim - np.mean(audio2_trim)) / (np.std(audio2_trim) + 1e-10)
    
    # Calculate correlation using the same method as cross-correlation
    correlation = np.sum(audio1_norm * audio2_norm) / len(audio1_norm)
    
    # Handle NaN case (silent audio)
    if np.isnan(correlation):
        correlation = 0.0
        
    return float(correlation)


def create_work_directory(base_path: str = "temp_pipeline") -> Path:
    """ Create and return working directory path """
    work_dir = Path(base_path)
    work_dir.mkdir(exist_ok=True)
    return work_dir