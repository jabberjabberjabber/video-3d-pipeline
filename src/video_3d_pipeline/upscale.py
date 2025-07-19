"""Simple depth upscaling using ffmpeg with NVENC."""

import ffmpeg
import argparse
from pathlib import Path
import os
import glob

from .utils import get_video_info


class SimpleDepthUpscaler:
    """ Simple depth upscaling using ffmpeg """
    
    def __init__(self, use_nvenc: bool = True):
        self.use_nvenc = use_nvenc
        
        print(f"Initializing Simple Depth Upscaler...")
        print(f"NVENC encoding: {self.use_nvenc}")
    
    def upscale_depth_maps_ffmpeg(self, 
                                  depth_dir: str, 
                                  target_width: int, 
                                  target_height: int,
                                  output_path: str,
                                  fps: float = 23.976):
        """ Upscale depth maps using ffmpeg """
        
        print(f"Processing depth upscaling with ffmpeg...")
        print(f"Input: {depth_dir}")
        print(f"Output: {output_path}")
        print(f"Target: {target_width}x{target_height} @ {fps}fps")
        
        # Find depth map files
        depth_files = sorted(glob.glob(os.path.join(depth_dir, "depth_*.png")))
        
        if not depth_files:
            raise ValueError(f"No depth maps found in {depth_dir}")
        
        print(f"Found {len(depth_files)} depth maps")
        
        # Create input pattern for ffmpeg
        input_pattern = os.path.join(depth_dir, "depth_%06d.png")
        
        try:
            # Build ffmpeg command
            stream = ffmpeg.input(input_pattern, r=fps, f='image2')
            
            # Scale to target resolution
            stream = ffmpeg.filter(stream, 'scale', target_width, target_height)
            
            # Choose encoder
            if self.use_nvenc:
                # NVENC HEVC encoding - better compression for depth data               
                stream = ffmpeg.output(stream, output_path, 
                                     vcodec='hevc_nvenc', 
                                     pix_fmt='yuv420p10le',  # 10-bit for better depth precision
                                     rc='vbr', 
                                     cq=20, 
                                     preset='p4', 
                                     r=fps)
            else:
                # CPU HEVC encoding fallback
                stream = ffmpeg.output(stream, output_path, 
                                     vcodec='libx265', 
                                     pix_fmt='yuv420p10le', 
                                     crf=20, 
                                     preset='medium', 
                                     r=fps)
                
            # Run ffmpeg
            print("Running ffmpeg...")
            ffmpeg.run(stream, overwrite_output=True, quiet=False)
            
        except ffmpeg.Error as e:
            print(f"FFmpeg error:")
            
            if e.stderr:
                print(e.stderr.decode())
                raise RuntimeError(f"FFmpeg processing failed: {e}")
                
        print(f"✓ Depth video saved: {output_path}")
        return output_path

    def upscale_depth_video_ffmpeg(self, 
                                   depth_video_path: str,
                                   target_width: int, 
                                   target_height: int,
                                   output_path: str,
                                   fps: float = 23.976):
        """ 	Upscale depth video directly using ffmpeg - more efficient than
				processing individual frames.
				
		"""
        
        print(f"Processing depth video upscaling with ffmpeg...")
        print(f"Input: {depth_video_path}")
        print(f"Output: {output_path}")
        print(f"Target: {target_width}x{target_height} @ {fps}fps")
        
        try:
            # Build ffmpeg command for video-to-video scaling
            stream = ffmpeg.input(depth_video_path)
            
            # Scale to target resolution
            stream = ffmpeg.filter(stream, 'scale', target_width, target_height)
            
            # Choose encoder
            if self.use_nvenc:
                # NVENC HEVC encoding - better compression for depth data                
                stream = ffmpeg.output(stream, output_path, 
                                     vcodec='hevc_nvenc', 
                                     pix_fmt='yuv420p10le',  # 10-bit for better depth precision
                                     rc='vbr', 
                                     cq=20, 
                                     preset='p4', 
                                     r=fps)
            else:
                # CPU HEVC encoding fallback
                stream = ffmpeg.output(stream, output_path, 
                                     vcodec='libx265', 
                                     pix_fmt='yuv420p10le', 
                                     crf=20, 
                                     preset='medium', 
                                     r=fps)
                
            # Run ffmpeg
            print("Running ffmpeg...")
            ffmpeg.run(stream, overwrite_output=True, quiet=False)
            
        except ffmpeg.Error as e:
            print(f"FFmpeg error:")
            
            if e.stderr:
                print(e.stderr.decode())
                raise RuntimeError(f"FFmpeg processing failed: {e}")
                
        print(f"✓ Upscaled depth video saved: {output_path}")
        return output_path

    def process_depth_upscaling(self, 
                               depth_dir: str,
                               video_4k_path: str,
                               output_path: str = None,
                               force_reprocess: bool = False) -> str:
        """ 	Main pipeline for depth upscaling - handles both individual 
				images and video input.

		"""
        
        print(f"Processing depth upscaling...")
        print(f"Depth input: {depth_dir}")
        print(f"4K video: {video_4k_path}")
        
        # Get target dimensions from 4K video
        video_info = get_video_info(video_4k_path)
        if not video_info:
            raise ValueError(f"Could not read video info: {video_4k_path}")
        
        target_width = video_info['width']
        target_height = video_info['height']
        fps = video_info['fps']
        
        print(f"Target resolution: {target_width}x{target_height} @ {fps}fps")
        
        # Generate output path if not provided
        if output_path is None:
            depth_dir_name = Path(depth_dir).name
            output_path = f"depth_4k_{depth_dir_name}.mp4"
        
        output_path = Path(output_path)
        
        # Check if already processed
        if output_path.exists() and not force_reprocess:
            print(f"✓ Using existing depth video: {output_path}")
            return str(output_path)
        
        # Determine input type: video file or directory with images
        depth_path = Path(depth_dir)
        
        # Check if it's a video file
        if depth_path.is_file() and depth_path.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']:
            print(f"Input is video file: {depth_path}")
            result = self.upscale_depth_video_ffmpeg(
                depth_video_path=str(depth_path),
                target_width=target_width,
                target_height=target_height,
                output_path=str(output_path),
                fps=fps
            )
        
        # Check if it's a directory or path to depth_video.mp4
        elif depth_path.is_dir():
            # Check for depth_video.mp4 in the directory (UniMatch output)
            depth_video_path = depth_path / "depth_video.mp4"
            
            if depth_video_path.exists():
                print(f"Found depth video in directory: {depth_video_path}")
                result = self.upscale_depth_video_ffmpeg(
                    depth_video_path=str(depth_video_path),
                    target_width=target_width,
                    target_height=target_height,
                    output_path=str(output_path),
                    fps=fps
                )
            else:
                # Fall back to individual image processing
                print(f"Looking for individual depth maps in directory: {depth_path}")
                result = self.upscale_depth_maps_ffmpeg(
                    depth_dir=str(depth_path),
                    target_width=target_width,
                    target_height=target_height,
                    output_path=str(output_path),
                    fps=fps
                )
        
        else:
            raise ValueError(f"Invalid depth input: {depth_dir}. Must be a video file or directory.")
        
        print(f"✓ Depth upscaling complete!")
        print(f"  Input: {depth_dir}")
        print(f"  Output: {result}")
        print(f"  Resolution: {target_width}x{target_height}")
        
        return result

    def convert_depth_format(self, 
                           input_video: str,
                           output_path: str = None,
                           force_reprocess: bool = False) -> str:
        """ 	Convert depth video format/encoding without changing resolution -
				used when depth is already at target resolution.

		"""
        
        print(f"Converting depth video format...")
        print(f"Input: {input_video}")
        
        if output_path is None:
            input_path = Path(input_video)
            output_path = f"depth_converted_{input_path.stem}.mp4"
        
        output_path = Path(output_path)
        
        # Check if already processed
        if output_path.exists() and not force_reprocess:
            print(f"✓ Using existing converted video: {output_path}")
            return str(output_path)
        
        # Get input video info
        try:
            video_info = get_video_info(input_video)
            if not video_info:
                raise ValueError(f"Could not read input video info: {input_video}")
            
            fps = video_info['fps']
            width = video_info['width']
            height = video_info['height']
            
            print(f"Converting: {width}x{height} @ {fps}fps")
            
            # Convert format without changing resolution
            stream = ffmpeg.input(input_video)
            
            # Choose encoder for optimal format
            if self.use_nvenc:
                # NVENC HEVC encoding - better compression for depth data                
                stream = ffmpeg.output(stream, str(output_path), 
                                     vcodec='hevc_nvenc', 
                                     pix_fmt='yuv420p10le',  # 10-bit for better depth precision
                                     rc='vbr', 
                                     cq=18,  # Slightly better quality for final output
                                     preset='p4', 
                                     r=fps)
            else:
                # CPU HEVC encoding fallback
                stream = ffmpeg.output(stream, str(output_path), 
                                     vcodec='libx265', 
                                     pix_fmt='yuv420p10le', 
                                     crf=18, 
                                     preset='medium', 
                                     r=fps)
            
            # Run ffmpeg
            print("Running format conversion...")
            ffmpeg.run(stream, overwrite_output=True, quiet=False)
            
        except ffmpeg.Error as e:
            print(f"FFmpeg error:")
            if e.stderr:
                print(e.stderr.decode())
            raise RuntimeError(f"Format conversion failed: {e}")
        
        print(f"✓ Format conversion complete!")
        print(f"  Input: {input_video}")
        print(f"  Output: {output_path}")
        print(f"  Resolution: {width}x{height} (unchanged)")
        
        return str(output_path)


def main():
    """ Command line interface for simple depth upscaling """
    parser = argparse.ArgumentParser(description='Simple depth upscaling using ffmpeg')
    parser.add_argument('depth_input', help='Directory containing depth maps or path to depth video')
    parser.add_argument('video_4k', help='Path to 4K 2D video (for dimensions)')
    parser.add_argument('--output', help='Output path for 4K depth video')
    parser.add_argument('--no-nvenc', action='store_true',
                       help='Disable NVENC, use CPU encoding')
    parser.add_argument('--force', action='store_true',
                       help='Force reprocessing even if output exists')
    
    args = parser.parse_args()
    
    try:
        # Initialize upscaler
        upscaler = SimpleDepthUpscaler(use_nvenc=not args.no_nvenc)
        
        # Process upscaling
        output_path = upscaler.process_depth_upscaling(
            depth_dir=args.depth_input,
            video_4k_path=args.video_4k,
            output_path=args.output,
            force_reprocess=args.force
        )
        
        print(f"\n✓ Success! 4K depth video: {output_path}")
        print(f"Ready for VisionDepth3D processing!")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
