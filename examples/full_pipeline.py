"""Example integration: alignment + depth extraction pipeline."""

import argparse
from pathlib import Path
import sys

from video_3d_pipeline import VideoAligner, HybridStereoDepthExtractor, GuidedDepthUpscaler


def run_full_pipeline(video_3d_path: str, 
                     video_4k_path: str,
                     duration: int = 60,
                     batch_size: int = 8,
                     model_checkpoint: str = "Intel/dpt-large"):
    """ Run complete alignment + depth extraction pipeline """
    
    print("=== Video 3D Pipeline: Alignment + Depth Extraction ===")
    print(f"3D Video: {video_3d_path}")
    print(f"4K Video: {video_4k_path}")
    print(f"Duration: {duration}s")
    print(f"Batch size: {batch_size}")
    print(f"Model: {model_checkpoint}")
    print()
    
    # Step 1: Temporal Alignment
    print("STEP 1: Temporal Alignment")
    print("-" * 40)
    
    try:
        aligner = VideoAligner(video_3d_path, video_4k_path, "temp_alignment")
        
        # Find audio alignment
        time_offset, correlation_strength = aligner.find_audio_alignment()
        
        if correlation_strength < 0.6:
            print("Warning: Low audio correlation. Continuing anyway...")
        
        # Extract aligned segments
        aligned_3d, aligned_4k = aligner.extract_aligned_segments(time_offset, duration)
        
        # Verify alignment quality
        final_score = aligner.verify_alignment(aligned_3d, aligned_4k)
        
        print(f"✓ Alignment complete!")
        print(f"  3D segment: {aligned_3d}")
        print(f"  4K segment: {aligned_4k}")
        print(f"  Final correlation: {final_score:.4f}")
        print()
        
    except Exception as e:
        print(f"✗ Alignment failed: {e}")
        return False
    
    # Step 2: Depth Extraction
    print("STEP 2: Depth Extraction from 3D Video")
    print("-" * 40)
    
    try:
        # Initialize depth extractor
        depth_extractor = HybridStereoDepthExtractor(
            model_checkpoint=model_checkpoint,
            work_dir="temp_depth",
            cache_dir="temp_depth",
            device="cuda",
            batch_size=batch_size,
            use_neural_guidance=True,
            stereo_only=False
        )
        
        # Process the aligned 3D video to extract depth maps
        depth_output_path = depth_extractor.process_video_sbs(
            video_path=aligned_3d,
            start_frame=0,
            max_frames=None,  # Process entire aligned segment
            force_reprocess=False
        )
        
        print(f"✓ Depth extraction complete!")
        print(f"  Depth maps: {depth_output_path}")
        print()
        
    except Exception as e:
        print(f"✗ Depth extraction failed: {e}")
        return False
    
    # Step 3: Depth Upscaling
    print("STEP 3: Depth Upscaling to 4K")
    print("-" * 40)
    
    try:
        # Initialize depth upscaler
        upscaler = GuidedDepthUpscaler(
            work_dir="temp_upscale",
            cache_dir="temp_upscale",
            device="cuda",
            upscale_method="guided",
            guided_filter_radius=8,
            guided_filter_eps=0.01
        )
        
        # Upscale depth maps to 4K using the 4K video as guidance
        depth_4k_video = upscaler.process_depth_upscaling(
            depth_dir=str(depth_output_path),
            video_4k_path=aligned_4k,
            output_path=None,  # Auto-generate path
            force_reprocess=False
        )
        
        print(f"✓ Depth upscaling complete!")
        print(f"  4K depth video: {depth_4k_video}")
        print()
        
    except Exception as e:
        print(f"✗ Depth upscaling failed: {e}")
        return False
    
    # Summary
    print("=== Pipeline Complete ===")
    print(f"✓ Aligned videos: temp_alignment/")
    print(f"✓ Depth maps: {depth_output_path}")
    print(f"✓ 4K depth video: {depth_4k_video}")
    print()
    print("Ready for VisionDepth3D:")
    print(f"1. Input 4K 2D video: {aligned_4k}")
    print(f"2. Input 4K depth video: {depth_4k_video}")
    print(f"3. VisionDepth3D will create 4K SBS 3D output")
    
    return True


def main():
    """ Command line interface for full pipeline """
    parser = argparse.ArgumentParser(description='Run alignment + depth extraction pipeline')
    parser.add_argument('video_3d', help='Path to 1080p SBS 3D video')
    parser.add_argument('video_4k', help='Path to 4K 2D video')
    parser.add_argument('--duration', type=int, default=60,
                       help='Duration of segments to process (seconds)')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Batch size for depth processing (default: 8)')
    parser.add_argument('--model', default="Intel/dpt-large",
                       help='Neural model checkpoint (Intel/dpt-large, Intel/dpt-hybrid-midas, etc.)')
    
    args = parser.parse_args()
    
    # Verify input files exist
    if not Path(args.video_3d).exists():
        print(f"Error: 3D video file not found: {args.video_3d}")
        return 1
    
    if not Path(args.video_4k).exists():
        print(f"Error: 4K video file not found: {args.video_4k}")
        return 1
    
    # Run pipeline
    success = run_full_pipeline(
        video_3d_path=args.video_3d,
        video_4k_path=args.video_4k,
        duration=args.duration,
        batch_size=args.batch_size,
        model_checkpoint=args.model
    )
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
