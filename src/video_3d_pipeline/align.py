"""Temporal alignment of videos with different resolutions."""

import cv2
import numpy as np
import ffmpeg
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Tuple

from .utils import get_video_info, extract_frames, create_work_directory, compute_frame_correlation


class VideoAligner:
    """ Handles temporal alignment between two videos of different resolutions """
    
    def __init__(self, video1_path: str, video2_path: str, work_dir: str = "temp_alignment"):
        self.video1_path = video1_path
        self.video2_path = video2_path
        self.work_dir = create_work_directory(work_dir)
        
        # Get video info
        self.video1_info = get_video_info(video1_path)
        self.video2_info = get_video_info(video2_path)
        
        if not self.video1_info or not self.video2_info:
            raise ValueError("Could not read video information")
        
        print(f"Video 1: {self.video1_info['width']}x{self.video1_info['height']} "
              f"@ {self.video1_info['fps']:.2f} fps, {self.video1_info['duration']:.1f}s")
        print(f"Video 2: {self.video2_info['width']}x{self.video2_info['height']} "
              f"@ {self.video2_info['fps']:.2f} fps, {self.video2_info['duration']:.1f}s")

    def find_best_alignment(self, frames1, timestamps1, frames2, timestamps2) -> Tuple[int, float]:
        """ Find best temporal alignment between two sets of keyframes """
        best_score = -1
        best_offset = 0
        scores = []
        offsets = []
        
        print("Finding temporal alignment...")
        
        # Try different offsets and compute correlation
        max_offset = min(len(frames1), len(frames2)) // 2
        
        for offset in range(-max_offset, max_offset):
            score = self._compute_alignment_score(frames1, frames2, offset)
            scores.append(score)
            offsets.append(offset)
            
            if score > best_score:
                best_score = score
                best_offset = offset
                
        print(f"Best alignment: offset={best_offset} frames, score={best_score:.4f}")
        
        # Plot alignment scores
        self._plot_alignment_scores(offsets, scores, best_offset)
        
        return best_offset, best_score

    def _compute_alignment_score(self, frames1, frames2, offset: int) -> float:
        """ Compute alignment score for given offset """
        if offset >= 0:
            f1_start, f2_start = offset, 0
        else:
            f1_start, f2_start = 0, -offset
            
        # Compare overlapping frames
        overlap_length = min(len(frames1) - f1_start, len(frames2) - f2_start, 20)
        
        if overlap_length < 5:  # Need minimum overlap
            return 0
            
        total_score = 0
        valid_comparisons = 0
        
        for i in range(overlap_length):
            f1_idx = f1_start + i
            f2_idx = f2_start + i
            
            if f1_idx < len(frames1) and f2_idx < len(frames2):
                corr = compute_frame_correlation(frames1[f1_idx], frames2[f2_idx])
                total_score += corr
                valid_comparisons += 1
                
        return total_score / valid_comparisons if valid_comparisons > 0 else 0

    def _plot_alignment_scores(self, offsets, scores, best_offset):
        """ Plot alignment scores and save to work directory """
        plt.figure(figsize=(12, 6))
        plt.plot(offsets, scores)
        plt.xlabel('Frame Offset')
        plt.ylabel('Alignment Score')
        plt.title('Temporal Alignment Scores')
        plt.axvline(x=best_offset, color='red', linestyle='--', 
                   label=f'Best offset: {best_offset}')
        plt.legend()
        plt.grid(True)
        plt.savefig(self.work_dir / 'alignment_scores.png', dpi=150, bbox_inches='tight')
        plt.close()

    def extract_aligned_segments(self, offset_frames: int, duration_seconds: int = 60) -> Tuple[str, str]:
        """ Extract aligned segments from both videos """
        # Convert frame offset to time offset
        fps1 = self.video1_info['fps']
        fps2 = self.video2_info['fps']
        
        # Find a good starting point (avoid beginning/end of videos)
        min_start_time = max(10, abs(offset_frames / fps1) if offset_frames < 0 else 0)
        max_end_time = min(self.video1_info['duration'], self.video2_info['duration']) - duration_seconds - 10
        
        if min_start_time >= max_end_time:
            raise ValueError("Videos too short or offset too large for requested duration")
            
        start_time = min_start_time
        
        print(f"Extracting {duration_seconds}s segments starting at {start_time:.1f}s...")
        
        # Extract from video1
        output1 = self.work_dir / f"video1_aligned_{int(start_time)}s.mp4"
        (
            ffmpeg
            .input(self.video1_path, ss=start_time, t=duration_seconds)
            .output(str(output1), vcodec='libx264', crf=18, preset='medium')
            .overwrite_output()
            .run(quiet=True)
        )
        
        # Extract from video2 with offset
        video2_start = start_time + (offset_frames / fps2)
        output2 = self.work_dir / f"video2_aligned_{int(start_time)}s.mp4"
        (
            ffmpeg
            .input(self.video2_path, ss=video2_start, t=duration_seconds)
            .output(str(output2), vcodec='libx264', crf=18, preset='medium')
            .overwrite_output()
            .run(quiet=True)
        )
        
        print(f"Aligned segments saved:")
        print(f"  Video 1: {output1}")
        print(f"  Video 2: {output2}")
        
        return str(output1), str(output2)

    def verify_alignment(self, video1_segment: str, video2_segment: str, num_frames: int = 10) -> float:
        """ Verify frame-by-frame alignment of extracted segments """
        cap1 = cv2.VideoCapture(video1_segment)
        cap2 = cv2.VideoCapture(video2_segment)
        
        frame_count1 = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_count2 = int(cap2.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Verifying alignment...")
        print(f"Segment 1 frames: {frame_count1}")
        print(f"Segment 2 frames: {frame_count2}")
        
        # Check frames at regular intervals
        verification_scores = []
        frame_indices = np.linspace(0, min(frame_count1, frame_count2) - 1, num_frames, dtype=int)
        
        for frame_idx in frame_indices:
            cap1.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            cap2.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            
            ret1, frame1 = cap1.read()
            ret2, frame2 = cap2.read()
            
            if ret1 and ret2:
                # Convert to grayscale
                gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
                gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
                
                corr = compute_frame_correlation(gray1, gray2)
                verification_scores.append(corr)
                
                print(f"Frame {frame_idx}: correlation = {corr:.4f}")
        
        cap1.release()
        cap2.release()
        
        avg_score = np.mean(verification_scores)
        print(f"\nAverage correlation: {avg_score:.4f}")
        
        if avg_score > 0.8:
            print("✓ Excellent alignment!")
        elif avg_score > 0.6:
            print("✓ Good alignment")
        elif avg_score > 0.4:
            print("⚠ Moderate alignment - may need adjustment")
        else:
            print("✗ Poor alignment - check source videos")
            
        return avg_score


def main():
    """ Command line interface for video alignment """
    parser = argparse.ArgumentParser(description='Temporally align two videos of different resolutions')
    parser.add_argument('video1', help='Path to first video')
    parser.add_argument('video2', help='Path to second video')
    parser.add_argument('--duration', type=int, default=60, 
                       help='Duration of segments to extract (seconds)')
    parser.add_argument('--work-dir', default='temp_alignment', 
                       help='Working directory for temporary files')
    parser.add_argument('--interval', type=float, default=1.0,
                       help='Keyframe extraction interval (seconds)')
    
    args = parser.parse_args()
    
    try:
        # Initialize aligner
        aligner = VideoAligner(args.video1, args.video2, args.work_dir)
        
        # Extract keyframes for alignment
        print("Extracting keyframes from video 1...")
        frames1, timestamps1 = extract_frames(args.video1, args.interval)
        
        print("Extracting keyframes from video 2...")  
        frames2, timestamps2 = extract_frames(args.video2, args.interval)
        
        # Find best alignment
        offset, score = aligner.find_best_alignment(frames1, timestamps1, frames2, timestamps2)
        
        if score < 0.3:
            print("Warning: Low alignment score. Videos may not be suitable for alignment.")
            return
        
        # Extract aligned segments
        seg1, seg2 = aligner.extract_aligned_segments(offset, args.duration)
        
        # Verify alignment
        final_score = aligner.verify_alignment(seg1, seg2)
        
        if final_score > 0.6:
            print(f"\n✓ Success! Aligned segments ready for 3D processing.")
        else:
            print(f"\n⚠ Alignment may need manual adjustment.")
            
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())