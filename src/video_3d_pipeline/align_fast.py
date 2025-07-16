"""Fast temporal alignment using audio-only correlation."""

import json
import argparse
from pathlib import Path
from typing import Tuple, Dict, Optional

from .utils import (get_video_info, create_work_directory, extract_audio, 
                   load_audio_for_sync, find_audio_offset, plot_audio_correlation,
                   verify_video_compatibility)


class FastVideoAligner:
    """Audio-only temporal alignment - no video re-encoding."""
    
    def __init__(self, video1_path: str, video2_path: str, work_dir: str = "temp_alignment"):
        self.video1_path = video1_path
        self.video2_path = video2_path
        self.work_dir = create_work_directory(work_dir)
        
        # Verify compatibility
        if not verify_video_compatibility(video1_path, video2_path):
            print("Warning: Videos may not be compatible for synchronization")
        
        # Get video info for frame precision calculations
        self.video1_info = get_video_info(video1_path)
        self.video2_info = get_video_info(video2_path)
        
        if not self.video1_info or not self.video2_info:
            raise ValueError("Could not read video information")
        
        print(f"Video 1: {self.video1_info['width']}x{self.video1_info['height']} "
              f"@ {self.video1_info['fps']:.2f} fps, {self.video1_info['duration']:.1f}s")
        print(f"Video 2: {self.video2_info['width']}x{self.video2_info['height']} "
              f"@ {self.video2_info['fps']:.2f} fps, {self.video2_info['duration']:.1f}s")

    def find_alignment(self, max_audio_length: float = 300) -> Dict:
        """Find temporal alignment and return offset data."""
        
        # Extract audio from both videos (cached)
        audio1_path = extract_audio(self.video1_path, self.work_dir, max_audio_length)
        audio2_path = extract_audio(self.video2_path, self.work_dir, max_audio_length)
        
        # Load audio waveforms
        audio1, sr1 = load_audio_for_sync(audio1_path, max_audio_length)
        audio2, sr2 = load_audio_for_sync(audio2_path, max_audio_length)
        
        if sr1 != sr2:
            print(f"Warning: Sample rate mismatch: {sr1} vs {sr2}")
        
        # Find optimal offset
        time_offset, correlation_strength = find_audio_offset(audio1, audio2, sr1)
        
        # Create visualization
        plot_audio_correlation(audio1, audio2, sr1, time_offset, self.work_dir)
        
        # Calculate frame precision metrics
        frame_duration = 1.0 / self.video1_info['fps']
        offset_frames = time_offset / frame_duration
        
        print(f"Audio alignment: {time_offset:.3f}s offset ({offset_frames:.1f} frames)")
        print(f"Correlation strength: {correlation_strength:.4f}")
        
        # Prepare alignment data
        alignment_data = {
            'video1_path': str(self.video1_path),
            'video2_path': str(self.video2_path),
            'time_offset_seconds': float(time_offset),
            'offset_frames': float(offset_frames),
            'correlation_strength': float(correlation_strength),
            'frame_duration': float(frame_duration),
            'video1_fps': self.video1_info['fps'],
            'video2_fps': self.video2_info['fps'],
            'sample_rate': int(sr1),
            'audio_length_analyzed': float(max_audio_length)
        }
        
        # Save alignment data for later use
        alignment_file = self.work_dir / 'alignment_data.json'
        with open(alignment_file, 'w') as f:
            json.dump(alignment_data, f, indent=2)
        
        print(f"Alignment data saved to: {alignment_file}")
        
        return alignment_data

    def assess_alignment_quality(self, alignment_data: Dict, tolerance_frames: float = 2.0) -> str:
        """Assess alignment quality and provide recommendations."""
        
        offset = alignment_data['time_offset_seconds']
        correlation = alignment_data['correlation_strength']
        frame_duration = alignment_data['frame_duration']
        
        precision_limit = frame_duration * tolerance_frames
        
        print(f"\nAlignment Assessment:")    
        print(f"Frame precision limit: ±{precision_limit:.3f}s ({tolerance_frames} frames)")
        
        if abs(offset) < precision_limit:
            quality = "EXCELLENT"
            print(f"✓ {quality}: Offset {offset:.3f}s is within frame precision")
            print("Videos are already well-aligned - no adjustment needed")
        elif correlation > 0.8:
            quality = "GOOD" 
            print(f"✓ {quality}: Strong correlation ({correlation:.3f})")
            print(f"Apply {offset:.3f}s offset in processing pipeline")
        elif correlation > 0.6:
            quality = "MODERATE"
            print(f"⚠ {quality}: Acceptable correlation ({correlation:.3f})")
            print(f"Apply {offset:.3f}s offset - verify results")
        else:
            quality = "POOR"
            print(f"✗ {quality}: Low correlation ({correlation:.3f})")
            print("Videos may not be from same source or need manual sync")
        
        return quality


def apply_offset_to_pipeline(alignment_file: str, target_video: str, 
                           output_path: str, start_time: float = 0, 
                           duration: Optional[float] = None) -> str:
    """Apply stored alignment offset during pipeline processing."""
    
    # Load alignment data
    with open(alignment_file, 'r') as f:
        alignment_data = json.load(f)
    
    offset = alignment_data['time_offset_seconds']
    
    # Determine which video needs offset applied
    if target_video == alignment_data['video1_path']:
        adjusted_start = start_time
        print(f"Video1 (reference): start at {adjusted_start:.3f}s")
    elif target_video == alignment_data['video2_path']:
        adjusted_start = start_time + offset
        print(f"Video2 (offset): start at {adjusted_start:.3f}s (original: {start_time:.3f}s + {offset:.3f}s offset)")
    else:
        raise ValueError(f"Target video {target_video} not found in alignment data")
    
    # Ensure adjusted start is not negative
    if adjusted_start < 0:
        print(f"Warning: Adjusted start time {adjusted_start:.3f}s < 0, using 0")
        adjusted_start = 0
    
    print(f"Use start_time={adjusted_start:.3f}s for {target_video}")
    
    return adjusted_start


def load_alignment_data(alignment_file: str) -> Dict:
    """Load previously computed alignment data."""
    
    alignment_path = Path(alignment_file)
    if not alignment_path.exists():
        raise FileNotFoundError(f"Alignment file not found: {alignment_file}")
    
    with open(alignment_path, 'r') as f:
        return json.load(f)


def main():
    """Command line interface for fast audio-only alignment."""
    parser = argparse.ArgumentParser(description='Fast audio-only video alignment')
    parser.add_argument('video1', help='Path to first video (reference)')
    parser.add_argument('video2', help='Path to second video (to be aligned)')
    parser.add_argument('--work-dir', default='temp_alignment', 
                       help='Working directory for temporary files')
    parser.add_argument('--max-audio', type=float, default=300.0,
                       help='Maximum audio length for analysis (seconds)')
    parser.add_argument('--tolerance', type=float, default=2.0,
                       help='Alignment tolerance in frame intervals')
    parser.add_argument('--min-correlation', type=float, default=0.6,
                       help='Minimum correlation to proceed')
    
    args = parser.parse_args()
    
    try:
        # Initialize fast aligner
        aligner = FastVideoAligner(args.video1, args.video2, args.work_dir)
        
        # Find alignment
        alignment_data = aligner.find_alignment(args.max_audio)
        
        # Assess quality
        quality = aligner.assess_alignment_quality(alignment_data, args.tolerance)
        
        # Check if correlation meets minimum threshold
        if alignment_data['correlation_strength'] < args.min_correlation:
            print(f"\nWarning: Correlation {alignment_data['correlation_strength']:.3f} below threshold {args.min_correlation}")
            response = input("Continue anyway? (y/n): ")
            if response.lower() != 'y':
                return 1
        
        print(f"\n✓ Alignment complete! Use alignment_data.json in pipeline steps.")
        print(f"Quality: {quality}")
        print(f"Offset: {alignment_data['time_offset_seconds']:.3f}s")
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
