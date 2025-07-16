"""Temporal alignment of videos using audio correlation."""

import cv2
import numpy as np
import ffmpeg
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Tuple

from .utils import (get_video_info, create_work_directory, extract_audio, 
                   load_audio_for_sync, find_audio_offset, plot_audio_correlation,
                   verify_video_compatibility)


class VideoAligner:
    """ Handles temporal alignment between two videos using audio correlation """
    
    def __init__(self, video1_path: str, video2_path: str, work_dir: str = "temp_alignment"):
        self.video1_path = video1_path
        self.video2_path = video2_path
        self.work_dir = create_work_directory(work_dir)
        
        # Verify video compatibility first
        if not verify_video_compatibility(video1_path, video2_path):
            print("Warning: Videos may not be compatible for accurate synchronization")
        
        # Get video info
        self.video1_info = get_video_info(video1_path)
        self.video2_info = get_video_info(video2_path)
        
        if not self.video1_info or not self.video2_info:
            raise ValueError("Could not read video information")
        
        print(f"Video 1: {self.video1_info['width']}x{self.video1_info['height']} "
              f"@ {self.video1_info['fps']:.2f} fps, {self.video1_info['duration']:.1f}s")
        print(f"Video 2: {self.video2_info['width']}x{self.video2_info['height']} "
              f"@ {self.video2_info['fps']:.2f} fps, {self.video2_info['duration']:.1f}s")

    def find_audio_alignment(self, max_audio_length: float = 300) -> Tuple[float, float]:
        """ Find best temporal alignment using audio cross-correlation """
        
        # Extract audio from both videos
        audio1_path = extract_audio(self.video1_path, self.work_dir, max_audio_length)
        audio2_path = extract_audio(self.video2_path, self.work_dir, max_audio_length)
        
        # Load audio waveforms
        audio1, sr1 = load_audio_for_sync(audio1_path, max_audio_length)
        audio2, sr2 = load_audio_for_sync(audio2_path, max_audio_length)
        
        # Verify sample rates match
        if sr1 != sr2:
            print(f"Warning: Sample rate mismatch: {sr1} vs {sr2}")
        
        # Find optimal offset using cross-correlation
        time_offset, correlation_strength = find_audio_offset(audio1, audio2, sr1)
        
        # Create visualization
        plot_audio_correlation(audio1, audio2, sr1, time_offset, self.work_dir)
        
        print(f"Audio alignment: {time_offset:.3f}s offset, strength: {correlation_strength:.4f}")
        
        return time_offset, correlation_strength

    def extract_aligned_segments(self, time_offset: float, duration_seconds: int = 60) -> Tuple[str, str]:
        """ Extract aligned segments from both videos using audio-derived offset """
        
        # Find a good starting point (avoid beginning/end of videos)
        min_duration = min(self.video1_info['duration'], self.video2_info['duration'])
        max_start_time = min_duration - duration_seconds - 10
        
        # Start from 10 seconds to avoid any intro/fade differences
        start_time = max(10, min(60, max_start_time / 2))  # Start partway through
        
        if start_time >= max_start_time:
            raise ValueError("Videos too short for requested duration")
        
        print(f"Extracting {duration_seconds}s segments starting at {start_time:.1f}s...")
        
        # Extract from video1 at start_time
        output1 = self.work_dir / f"video1_aligned_{int(start_time)}s.mp4"
        (
            ffmpeg
            .input(self.video1_path, ss=start_time, t=duration_seconds)
            .output(str(output1), vcodec='libx264', crf=18, preset='medium')
            .overwrite_output()
            .run(quiet=True)
        )
        
        # Extract from video2 with time offset applied
        video2_start = start_time - time_offset  # Apply the offset
        if video2_start < 0:
            video2_start = 0
            print(f"Warning: Adjusted video2 start time to 0 (was {video2_start:.1f}s)")
        
        output2 = self.work_dir / f"video2_aligned_{int(start_time)}s.mp4"
        (
            ffmpeg
            .input(self.video2_path, ss=video2_start, t=duration_seconds)
            .output(str(output2), vcodec='libx264', crf=18, preset='medium')
            .overwrite_output()
            .run(quiet=True)
        )
        
        print(f"Aligned segments saved:")
        print(f"  Video 1: {output1} (start: {start_time:.1f}s)")
        print(f"  Video 2: {output2} (start: {video2_start:.1f}s)")
        
        return str(output1), str(output2)

    def verify_alignment(self, video1_segment: str, video2_segment: str, num_frames: int = 10) -> float:
        """ Verify alignment quality by checking audio correlation of segments """
        
        # Extract audio from both segments
        seg1_audio_path = extract_audio(video1_segment, self.work_dir, 60)
        seg2_audio_path = extract_audio(video2_segment, self.work_dir, 60)
        
        # Load audio
        audio1, sr1 = load_audio_for_sync(seg1_audio_path, 60)
        audio2, sr2 = load_audio_for_sync(seg2_audio_path, 60)
        
        # Check correlation at zero offset (should be high if aligned)
        min_len = min(len(audio1), len(audio2))
        audio1_trim = audio1[:min_len]
        audio2_trim = audio2[:min_len]
        
        # Compute correlation coefficient
        correlation = np.corrcoef(audio1_trim, audio2_trim)[0, 1]
        
        print(f"Segment audio correlation: {correlation:.4f}")
        
        if correlation > 0.8:
            print("✓ Excellent alignment!")
        elif correlation > 0.6:
            print("✓ Good alignment")
        elif correlation > 0.4:
            print("⚠ Moderate alignment - may need adjustment")
        else:
            print("✗ Poor alignment - check source videos")
        
        return correlation


def main():
    """ Command line interface for video alignment using audio correlation """
    parser = argparse.ArgumentParser(description='Align two videos using audio correlation')
    parser.add_argument('video1', help='Path to first video')
    parser.add_argument('video2', help='Path to second video')
    parser.add_argument('--duration', type=int, default=60, 
                       help='Duration of segments to extract (seconds)')
    parser.add_argument('--work-dir', default='temp_alignment', 
                       help='Working directory for temporary files')
    parser.add_argument('--max-audio', type=float, default=300.0,
                       help='Maximum audio length for sync analysis (seconds)')
    
    args = parser.parse_args()
    
    try:
        # Initialize aligner
        aligner = VideoAligner(args.video1, args.video2, args.work_dir)
        
        # Find alignment using audio correlation
        time_offset, correlation_strength = aligner.find_audio_alignment(args.max_audio)
        
        if correlation_strength < 0.3:
            print("Warning: Low audio correlation. Videos may not be from same source.")
            response = input("Continue anyway? (y/n): ")
            if response.lower() != 'y':
                return 1
        
        # Extract aligned segments
        seg1, seg2 = aligner.extract_aligned_segments(time_offset, args.duration)
        
        # Verify alignment quality
        final_score = aligner.verify_alignment(seg1, seg2)
        
        if final_score > 0.6:
            print(f"\n✓ Success! Aligned segments ready for 3D processing.")
            print(f"Audio correlation: {correlation_strength:.4f}")
            print(f"Time offset: {time_offset:.3f}s")
        else:
            print(f"\n⚠ Alignment may need manual adjustment.")
            print(f"Consider using different source videos or manual sync points.")
            
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())