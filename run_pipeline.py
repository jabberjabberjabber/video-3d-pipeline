"""Optimized 3D video pipeline runner."""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from video_3d_pipeline.align import VideoAligner
from video_3d_pipeline.upscale import SimpleDepthUpscaler
from video_3d_pipeline.unimatch_depth import UniMatchStereoDepthExtractor
from video_3d_pipeline.utils import get_video_info


def run_pipeline(sbs_video: str, 
                video_4k: str,
                work_dir: str = "temp_pipeline",
                start_frame: int = 0,
                max_frames: int = None,
                skip_alignment: bool = False,
                skip_depth: bool = False,
                skip_upscale: bool = False,
                force_reprocess: bool = False):
    """Run the complete optimized pipeline."""
    
    print("=== OPTIMIZED 3D VIDEO PIPELINE ===")
    print(f"SBS 1080p: {sbs_video}")
    print(f"4K video: {video_4k}")
    print(f"Work dir: {work_dir}")
    print(f"Start frame: {start_frame}")
    if max_frames:
        print(f"Max frames: {max_frames}")
    
    total_start = time.time()
    results = {}
    
    if not skip_alignment:
        print("\n--- Step 1: Audio-Only Alignment ---")
        step_start = time.time()
        
        aligner = VideoAligner(sbs_video, video_4k, work_dir)
        alignment_data = aligner.find_alignment(max_audio_length=300)
        quality = aligner.assess_alignment_quality(alignment_data)
        
        results['alignment'] = {
            'time': time.time() - step_start,
            'offset': alignment_data['time_offset_seconds'],
            'quality': quality,
            'data_file': f"{work_dir}/alignment_data.json"
        }
        
        print(f"✓ Alignment: {results['alignment']['time']:.1f}s")
        print(f"  Offset: {results['alignment']['offset']:.3f}s")
        print(f"  Quality: {results['alignment']['quality']}")
    else:
        print("\nSkipping alignment step")
     
    if not skip_depth:
        print("\n--- Step 2: Direct 4K Depth Extraction ---")
        step_start = time.time()
        
        # Get 4K video dimensions for direct extraction
        video_4k_info = get_video_info(video_4k)
        if not video_4k_info:
            raise ValueError(f"Could not read 4K video info: {video_4k}")
        
        target_width = video_4k_info['width']
        target_height = video_4k_info['height'] 
        target_fps = video_4k_info['fps']
        
        print(f"4K target: {target_width}x{target_height} @ {target_fps:.1f}fps")
        
        # Create extractor with 4K output dimensions
        extractor = UniMatchStereoDepthExtractor(
            work_dir=work_dir,
            cache_dir=work_dir,
            unsqueeze_sbs=True,
            batch_size=4,
            inference_size=[480, 854],  # Keep fast inference, upscale at output
            num_reg_refine=1, 
            num_scales=1,
            target_output_size=[target_height, target_width],  # Direct 4K output
            target_fps=target_fps
        )
        
        depth_video_path = extractor.process_video_sbs(
            video_path=sbs_video,
            start_frame=start_frame,
            max_frames=max_frames,
            force_reprocess=force_reprocess
        )
        
        results['depth'] = {
            'time': time.time() - step_start,
            'output_path': str(depth_video_path),
            'resolution': f"{target_width}x{target_height}",
            'direct_4k': True
        }
        
        print(f"✓ Direct 4K depth extraction: {results['depth']['time']:.1f}s")
        print(f"  Output: {results['depth']['output_path']}")
        print(f"  Resolution: {results['depth']['resolution']}")
    else:
        print("\nSkipping depth extraction step")
        results['depth'] = {
            'time': 0,
            'output_path': f"{work_dir}/depth_video.mp4",
            'direct_4k': False
        }
    
    # Upscaling step is now optional - only for format conversion or quality adjustments
    if not skip_upscale:
        print("\n--- Step 3: Optional Format Conversion ---")
        step_start = time.time()
        
        if results['depth'].get('direct_4k', False):
            print("Depth already at 4K - converting format/encoding only")
            
            upscaler = SimpleDepthUpscaler(use_nvenc=True)
            
            # Just convert format without changing resolution
            final_output = upscaler.convert_depth_format(
                input_video=results['depth']['output_path'],
                output_path=f"{work_dir}/depth_final.mp4",
                force_reprocess=force_reprocess
            )
        else:
            print("Legacy upscaling mode - scaling resolution")
            
            upscaler = SimpleDepthUpscaler(use_nvenc=True)
            final_output = upscaler.process_depth_upscaling(
                depth_dir=results['depth']['output_path'],
                video_4k_path=video_4k,
                output_path=f"{work_dir}/depth_final.mp4",
                force_reprocess=force_reprocess
            )
        
        results['upscale'] = {
            'time': time.time() - step_start,
            'output_video': final_output,
            'operation': 'format_conversion' if results['depth'].get('direct_4k') else 'upscaling'
        }
        
        print(f"✓ {results['upscale']['operation'].title()}: {results['upscale']['time']:.1f}s")
        print(f"  Output: {results['upscale']['output_video']}")
    else:
        print("\nSkipping format conversion step")
        results['upscale'] = {
            'time': 0,
            'output_video': results['depth']['output_path'],
            'operation': 'skipped'
        }

    
    total_time = time.time() - total_start
    print(f"\n=== PIPELINE COMPLETE ===")
    print(f"Total time: {total_time:.1f}s")
    
    for step, data in results.items():
        print(f"  {step.capitalize()}: {data['time']:.1f}s")
    
    print(f"\nFinal outputs:")
    print(f"✓ 4K RGB video: {video_4k}")
    print(f"✓ 4K depth video: {results['upscale']['output_video']}")
    
    if results['depth'].get('direct_4k'):
        print("✅ Optimized: Depth extracted directly at 4K (no upscaling needed)")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Optimized 3D video pipeline')
    parser.add_argument('sbs_video', help='Path to SBS 1080p video')
    parser.add_argument('video_4k', help='Path to 4K 2D video')
    parser.add_argument('--work-dir', default='temp_pipeline',
                       help='Working directory (default: temp_pipeline)')
    parser.add_argument('--start-frame', type=int, default=0,
                       help='Starting frame number (default: 0)')
    parser.add_argument('--max-frames', type=int,
                       help='Maximum frames to process (for testing)')
    parser.add_argument('--skip-alignment', action='store_true',
                       help='Skip alignment step')
    parser.add_argument('--skip-depth', action='store_true', 
                       help='Skip depth extraction step')
    parser.add_argument('--skip-upscale', action='store_true',
                       help='Skip format conversion step (use raw depth video)')
    parser.add_argument('--force', action='store_true',
                       help='Force reprocessing of all steps')
    
    args = parser.parse_args()
    
    try:
        results = run_pipeline(
            sbs_video=args.sbs_video,
            video_4k=args.video_4k,
            work_dir=args.work_dir,
            start_frame=args.start_frame,
            max_frames=args.max_frames,
            skip_alignment=args.skip_alignment,
            skip_depth=args.skip_depth,
            skip_upscale=args.skip_upscale,
            force_reprocess=args.force
        )
        
        print("\n🎉 Pipeline completed successfully!")
        return 0
        
    except Exception as e:
        print(f"\n💥 Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
