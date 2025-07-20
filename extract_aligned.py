"""Extract synchronized video segments using pre-computed alignment data."""

import json
import argparse
import subprocess
import sys
from pathlib import Path
from typing import Dict, Tuple, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from video_3d_pipeline.align import VideoAligner
from video_3d_pipeline.utils import get_video_info

class AlignedExtractor:
    """Extract synchronized video segments using alignment data."""
    
    def __init__(self, alignment_file: str):
        self.alignment_data = self._load_alignment_data(alignment_file)
        self.video1_path = self.alignment_data['video1_path']
        self.video2_path = self.alignment_data['video2_path']
        self.time_offset = self.alignment_data['time_offset_seconds']
        self.video1_fps = self.alignment_data['video1_fps']
        self.video2_fps = self.alignment_data['video2_fps']
        
        # Verify videos still exist
        if not Path(self.video1_path).exists():
            raise FileNotFoundError(f"Video1 not found: {self.video1_path}")
        if not Path(self.video2_path).exists():
            raise FileNotFoundError(f"Video2 not found: {self.video2_path}")
        
        print(f"Loaded alignment data:")
        print(f"  Video1: {self.video1_path}")
        print(f"  Video2: {self.video2_path}")
        print(f"  Time offset: {self.time_offset:.3f}s")
        print(f"  Correlation: {self.alignment_data['correlation_strength']:.4f}")

    def _load_alignment_data(self, alignment_file: str) -> Dict:
        """Load alignment data from JSON file."""
        alignment_path = Path(alignment_file)
        if not alignment_path.exists():
            raise FileNotFoundError(f"Alignment file not found: {alignment_file}")
        
        with open(alignment_path, 'r') as f:
            return json.load(f)

    def _frame_to_seconds(self, frame_number: int, fps: float) -> float:
        """Convert frame number to seconds."""
        return frame_number / fps

    def _seconds_to_frame(self, seconds: float, fps: float) -> int:
        """Convert seconds to frame number."""
        return int(seconds * fps)

    def calculate_extraction_times(self, start_frame: int, duration: float) -> Tuple[float, float]:
        """Calculate extraction start times for both videos."""
        
        # Convert start frame to seconds using video1 fps (reference)
        start_seconds = self._frame_to_seconds(start_frame, self.video1_fps)
        
        # Video1 extraction time (reference)
        video1_start = start_seconds
        
        # Video2 extraction time (apply offset)
        video2_start = start_seconds + self.time_offset
        
        # Ensure start times are not negative
        if video1_start < 0:
            print(f"Warning: Video1 start time {video1_start:.3f}s < 0, using 0")
            video1_start = 0
        
        if video2_start < 0:
            print(f"Warning: Video2 start time {video2_start:.3f}s < 0, using 0")
            video2_start = 0
        
        print(f"Extraction times:")
        print(f"  Video1: {video1_start:.3f}s (frame {start_frame})")
        print(f"  Video2: {video2_start:.3f}s (frame {self._seconds_to_frame(video2_start, self.video2_fps)})")
        
        return video1_start, video2_start

    def extract_segment(self, start_frame: int, duration: float, 
                       output_dir: str = "extracted_segments",
                       output_prefix: str = "aligned") -> Tuple[str, str]:
        """Extract synchronized video segments from both videos."""
        
        # Calculate extraction times
        video1_start, video2_start = self.calculate_extraction_times(start_frame, duration)
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Generate output filenames
        video1_stem = Path(self.video1_path).stem
        video2_stem = Path(self.video2_path).stem
        
        output1 = output_path / f"{output_prefix}_{video1_stem}_f{start_frame:06d}.mp4"
        output2 = output_path / f"{output_prefix}_{video2_stem}_f{start_frame:06d}.mp4"
        
        print(f"Extracting {duration}s segments...")
        
        # Extract from video1
        self._extract_video_segment(
            self.video1_path, video1_start, duration, str(output1)
        )
        
        # Extract from video2  
        self._extract_video_segment(
            self.video2_path, video2_start, duration, str(output2)
        )
        
        print(f"Extracted segments:")
        print(f"  Video1: {output1}")
        print(f"  Video2: {output2}")
        
        return str(output1), str(output2)

    def _extract_video_segment(self, input_path: str, start_time: float, 
                              duration: float, output_path: str) -> None:
        """Extract a video segment using ffmpeg."""
        
        cmd = [
            'ffmpeg',
            '-i', input_path,
            '-ss', str(start_time),
            '-t', str(duration),
            '-c', 'copy',  # Stream copy for speed
            '-avoid_negative_ts', 'make_zero',
            '-y',  # Overwrite output file
            output_path
        ]
        
        print(f"Extracting: {Path(input_path).name} -> {Path(output_path).name}")
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(f"  ✓ Success")
        except subprocess.CalledProcessError as e:
            print(f"  ✗ FFmpeg error: {e}")
            print(f"  Command: {' '.join(cmd)}")
            if e.stderr:
                print(f"  Error output: {e.stderr}")
            raise

    def extract_multiple_segments(self, segments: list, output_dir: str = "extracted_segments",
                                 output_prefix: str = "aligned") -> list:
        """Extract multiple synchronized segments."""
        
        results = []
        
        for i, (start_frame, duration) in enumerate(segments):
            print(f"\nExtracting segment {i+1}/{len(segments)}")
            
            # Use segment number in filename
            segment_prefix = f"{output_prefix}_seg{i+1:03d}"
            
            output1, output2 = self.extract_segment(
                start_frame, duration, output_dir, segment_prefix
            )
            
            results.append({
                'segment_number': i + 1,
                'start_frame': start_frame,
                'duration': duration,
                'video1_output': output1,
                'video2_output': output2
            })
        
        return results

    def get_video_info(self) -> Dict:
        """Get information about both videos."""
        
        video1_info = get_video_info(self.video1_path)
        video2_info = get_video_info(self.video2_path)
        
        return {
            'video1': video1_info,
            'video2': video2_info,
            'alignment': self.alignment_data
        }

    def validate_extraction_request(self, start_frame: int, duration: float) -> bool:
        """Validate that extraction request is within video bounds."""
        
        # Get video information
        video1_info = get_video_info(self.video1_path)
        video2_info = get_video_info(self.video2_path)
        
        if not video1_info or not video2_info:
            print("Warning: Could not validate video information")
            return True  # Allow extraction to proceed
        
        # Calculate extraction times
        video1_start, video2_start = self.calculate_extraction_times(start_frame, duration)
        
        # Check bounds
        video1_end = video1_start + duration
        video2_end = video2_start + duration
        
        issues = []
        
        if video1_end > video1_info['duration']:
            issues.append(f"Video1 extraction ({video1_end:.1f}s) exceeds duration ({video1_info['duration']:.1f}s)")
        
        if video2_end > video2_info['duration']:
            issues.append(f"Video2 extraction ({video2_end:.1f}s) exceeds duration ({video2_info['duration']:.1f}s)")
        
        if issues:
            print("Validation warnings:")
            for issue in issues:
                print(f"  ⚠ {issue}")
            return False
        
        print("✓ Extraction request validated")
        return True


def main():
    """Command line interface for extracting aligned video segments."""
    
    parser = argparse.ArgumentParser(
        description='Extract synchronized video segments using alignment data'
    )
    parser.add_argument('video1', help='Path to first video (reference)')
    parser.add_argument('video2', help='Path to second video (aligned)')
    parser.add_argument('--json', required=True, 
                       help='Path to alignment data JSON file')
    parser.add_argument('--start-frame', type=int, required=True,
                       help='Starting frame number (reference to video1)')
    parser.add_argument('--duration', type=float, required=True,
                       help='Duration in seconds to extract')
    parser.add_argument('--output-dir', default='extracted_segments',
                       help='Output directory for extracted segments')
    parser.add_argument('--output-prefix', default='aligned',
                       help='Prefix for output filenames')
    parser.add_argument('--validate', action='store_true',
                       help='Validate extraction bounds before processing')
    parser.add_argument('--info', action='store_true',
                       help='Show video and alignment information only')
    
    args = parser.parse_args()
    
    try:
        # Initialize extractor
        extractor = AlignedExtractor(args.json)
        
        # Verify video paths match alignment data
        if args.video1 != extractor.video1_path:
            print(f"Warning: Video1 path mismatch")
            print(f"  Command line: {args.video1}")
            print(f"  Alignment data: {extractor.video1_path}")
        
        if args.video2 != extractor.video2_path:
            print(f"Warning: Video2 path mismatch")
            print(f"  Command line: {args.video2}")
            print(f"  Alignment data: {extractor.video2_path}")
        
        # Show info and exit if requested
        if args.info:
            info = extractor.get_video_info()
            print(f"\nVideo Information:")
            print(f"Video1: {info['video1']['width']}x{info['video1']['height']} "
                  f"@ {info['video1']['fps']:.2f} fps, {info['video1']['duration']:.1f}s")
            print(f"Video2: {info['video2']['width']}x{info['video2']['height']} "
                  f"@ {info['video2']['fps']:.2f} fps, {info['video2']['duration']:.1f}s")
            return 0
        
        # Validate extraction request
        if args.validate:
            if not extractor.validate_extraction_request(args.start_frame, args.duration):
                response = input("Continue with extraction? (y/n): ")
                if response.lower() != 'y':
                    return 1
        
        # Extract segments
        output1, output2 = extractor.extract_segment(
            args.start_frame, args.duration, args.output_dir, args.output_prefix
        )
        
        print(f"\n✓ Extraction complete!")
        print(f"Synchronized segments saved to: {args.output_dir}")
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
