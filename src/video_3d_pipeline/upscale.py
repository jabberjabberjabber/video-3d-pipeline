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
            
            # NVENC GPU encoding                
                stream = ffmpeg.output(stream, output_path, vcodec='h264_nvenc', pix_fmt='yuv420p', crf=18, preset='medium', r=fps)
            else:
                # CPU encoding fallback
                stream = ffmpeg.output(stream, output_path, vcodec='libx264', pix_fmt='yuv420p', crf=18, preset='medium', r=fps)
                
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

    def process_depth_upscaling(self, 
                               depth_dir: str,
                               video_4k_path: str,
                               output_path: str = None,
                               force_reprocess: bool = False) -> str:
        """ Main pipeline for depth upscaling """
        
        print(f"Processing depth upscaling...")
        print(f"Depth maps: {depth_dir}")
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
        
        # Process with ffmpeg
        result = self.upscale_depth_maps_ffmpeg(
            depth_dir=depth_dir,
            target_width=target_width,
            target_height=target_height,
            output_path=str(output_path),
            fps=fps
        )
        
        print(f"✓ Depth upscaling complete!")
        print(f"  Input: {depth_dir}")
        print(f"  Output: {result}")
        print(f"  Resolution: {target_width}x{target_height}")
        
        return result


def main():
    """ Command line interface for simple depth upscaling """
    parser = argparse.ArgumentParser(description='Simple depth upscaling using ffmpeg')
    parser.add_argument('depth_dir', help='Directory containing depth maps')
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
            depth_dir=args.depth_dir,
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
