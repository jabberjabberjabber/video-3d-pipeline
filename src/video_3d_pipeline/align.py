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
                   verify_video_compatibility, calculate_audio_correlation)


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
        print(f"Time offset found: {time_offset:.3f}s")
        
        # Apply offset correctly:
        # If time_offset is negative, video1 is ahead of video2
        # So we need to start video2 extraction earlier (at a smaller timestamp)
        # If time_offset is positive, video1 is behind video2  
        # So we need to start video2 extraction later (at a larger timestamp)
        video2_start = start_time + time_offset
        
        if video2_start < 0:
            video2_start = 0
            print(f"Warning: Adjusted video2 start time to 0 (was {video2_start:.1f}s)")
        
        print(f"Video1 segment: {start_time:.3f}s to {start_time + duration_seconds:.3f}s")
        print(f"Video2 segment: {video2_start:.3f}s to {video2_start + duration_seconds:.3f}s")
        
        # Extract from video1 - use simple, accurate extraction
        output1 = self.work_dir / f"video1_aligned_{duration_seconds}s.mp4"
        print(f"Extracting video1 from {start_time:.3f}s...")
        
        try:
            (
                ffmpeg
                .input(self.video1_path)
                .output(str(output1), 
                       ss=start_time,  # Precise seek
                       t=duration_seconds,
                       vcodec='libx264', 
                       acodec='copy',  # Copy audio without re-encoding
                       crf=18, 
                       preset='medium',
                       avoid_negative_ts='make_zero')
                .overwrite_output()
                .run(quiet=False, capture_stdout=True, capture_stderr=True)
            )
        except ffmpeg.Error as e:
            print(f"FFmpeg error extracting video1:")
            print(f"stderr: {e.stderr.decode() if e.stderr else 'None'}")
            raise
        
        # Extract from video2 - use simple, accurate extraction
        output2 = self.work_dir / f"video2_aligned_{duration_seconds}s.mp4"
        print(f"Extracting video2 from {video2_start:.3f}s...")
        
        try:
            (
                ffmpeg
                .input(self.video2_path)
                .output(str(output2), 
                       ss=video2_start,  # Precise seek
                       t=duration_seconds,
                       vcodec='libx264', 
                       acodec='copy',  # Copy audio without re-encoding
                       crf=18, 
                       preset='medium',
                       avoid_negative_ts='make_zero')
                .overwrite_output()
                .run(quiet=False, capture_stdout=True, capture_stderr=True)
            )
        except ffmpeg.Error as e:
            print(f"FFmpeg error extracting video2:")
            print(f"stderr: {e.stderr.decode() if e.stderr else 'None'}")
            raise
        
        print(f"Aligned segments saved:")
        print(f"  Video 1: {output1} (start: {start_time:.3f}s)")
        print(f"  Video 2: {output2} (start: {video2_start:.3f}s)")
        
        return str(output1), str(output2)

    def verify_extraction_timing(self, video1_segment: str, video2_segment: str) -> None:
        """ Verify that the extracted segments have the correct timing """
        
        print("Verifying extraction timing...")
        
        # Get detailed info about the extracted segments
        try:
            # Check video1 segment
            probe1 = ffmpeg.probe(video1_segment)
            v1_duration = float(probe1['format']['duration'])
            v1_start_time = float(probe1['format'].get('start_time', 0))
            
            # Check video2 segment  
            probe2 = ffmpeg.probe(video2_segment)
            v2_duration = float(probe2['format']['duration'])
            v2_start_time = float(probe2['format'].get('start_time', 0))
            
            print(f"Video1 segment: duration={v1_duration:.3f}s, start_time={v1_start_time:.3f}s")
            print(f"Video2 segment: duration={v2_duration:.3f}s, start_time={v2_start_time:.3f}s")
            
            # Check if durations match
            duration_diff = abs(v1_duration - v2_duration)
            if duration_diff > 0.1:  # More than 100ms difference
                print(f"Warning: Duration mismatch: {duration_diff:.3f}s")
            else:
                print(f"✓ Durations match within {duration_diff:.3f}s")
                
        except Exception as e:
            print(f"Error verifying extraction timing: {e}")

    def verify_alignment(self, video1_segment: str, video2_segment: str, tolerance_frames: float = 2.0) -> float:
        """ Verify alignment quality by checking audio correlation of segments """
        
        # Extract audio from both segments
        seg1_audio_path = extract_audio(video1_segment, self.work_dir, 60)
        seg2_audio_path = extract_audio(video2_segment, self.work_dir, 60)
        
        # Load audio
        audio1, sr1 = load_audio_for_sync(seg1_audio_path, 60)
        audio2, sr2 = load_audio_for_sync(seg2_audio_path, 60)
        
        # Use the same correlation calculation as the main alignment
        correlation = calculate_audio_correlation(audio1, audio2)
        
        # Also check if there's still a small offset (within frame precision)
        frame_duration = 1.0 / self.video1_info['fps']
        precision_limit = frame_duration * tolerance_frames
        
        # Quick offset check to see if we're within tolerance
        offset, _ = find_audio_offset(audio1, audio2, sr1)
        offset_within_tolerance = abs(offset) < precision_limit
        
        #print(f"Segment audio correlation: {correlation:.4f}")
        print(f"Residual offset: {offset:.3f}s (tolerance: {precision_limit:.3f}s)")
        
        # Adjust thresholds based on precision limits
        #if correlation > 0.8 and offset_within_tolerance:
        if offset_within_tolerance:
            #print("✓ Excellent alignment!")
        #elif correlation > 0.6 and offset_within_tolerance:
            print("✓ Good alignment (within frame precision)")
        #elif correlation > 0.4:
            #print("⚠ Moderate alignment - may need adjustment")
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
    parser.add_argument('--tolerance', type=float, default=2.0,
                       help='Alignment tolerance in frame intervals (default: 2 frames)')
    
    args = parser.parse_args()
    
    try:
        # Initialize aligner
        aligner = VideoAligner(args.video1, args.video2, args.work_dir)
        
        # Find alignment using audio correlation
        time_offset, correlation_strength = aligner.find_audio_alignment(args.max_audio)
        
        # Calculate frame-level precision limit
        frame_duration = 1.0 / aligner.video1_info['fps']
        precision_limit = frame_duration * args.tolerance  # Use configurable tolerance
        
        print(f"Frame precision limit: {precision_limit:.3f}s ({precision_limit*1000:.1f}ms)")
        
        # Check if offset is within acceptable precision
        if abs(time_offset) < precision_limit:
            print(f"✓ Offset {time_offset:.3f}s is within frame precision tolerance")
            print(f"Videos are already well-aligned (within {precision_limit*1000:.1f}ms)")
            return 0
        
        if correlation_strength < 0.6:
            print("Warning: Low audio correlation. Videos may not be from same source.")
            response = input("Continue anyway? (y/n): ")
            if response.lower() != 'y':
                return 1
        
        # Extract aligned segments
        seg1, seg2 = aligner.extract_aligned_segments(time_offset, args.duration)
        
        # Verify extraction timing
        aligner.verify_extraction_timing(seg1, seg2)
        
        # Verify alignment quality
        final_score = aligner.verify_alignment(seg1, seg2, args.tolerance)
        
        # Calculate frame-level precision for success criteria
        '''
        frame_duration = 1.0 / aligner.video1_info['fps']
        precision_limit = frame_duration * args.tolerance
        
        # Check final alignment success with realistic thresholds
        if final_score > 0.6:  # Lower threshold since we account for frame precision
            print(f"\n✓ Success! Aligned segments ready for 3D processing.")
            print(f"Audio correlation: {correlation_strength:.4f}")
            print(f"Time offset: {time_offset:.3f}s")
            print(f"Final correlation: {final_score:.4f}")
        else:
            print(f"\n⚠ Alignment may need manual adjustment.")
            print(f"Consider using different source videos or manual sync points.")
            print(f"Final correlation score: {final_score:.4f}")
            print(f"Note: Video alignment precision is limited to ~{precision_limit*1000:.1f}ms (frame boundaries)")
        '''    
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())