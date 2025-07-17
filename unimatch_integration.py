#!/usr/bin/env python3
"""
Integration example showing how to replace IGEV with UniMatch for faster depth extraction.
This script demonstrates the new UniMatch-based pipeline with optional temporal consistency.
"""

import argparse
import sys
from pathlib import Path

# Import the new UniMatch modules
from video_3d_pipeline.unimatch_depth import UniMatchStereoDepthExtractor
from video_3d_pipeline.unimatch_flow_temporal import UniMatchFlowProcessor, TemporalDepthEnhancer


def benchmark_comparison(video_path: str, max_frames: int = 50):
    """ Compare IGEV vs UniMatch performance """
    import time
    
    print("=== PERFORMANCE COMPARISON ===")
    print(f"Testing with {max_frames} frames from: {video_path}")
    
    # Test IGEV (original)
    try:
        from video_3d_pipeline.depth import IGEVStereoDepthExtractor
        print("\n--- IGEV Baseline ---")
        
        igev_extractor = IGEVStereoDepthExtractor(
            batch_size=4,  # Smaller batch for fair comparison
            valid_iters=16  # Reduced iterations for speed
        )
        
        start_time = time.time()
        igev_output = igev_extractor.process_video_sbs(
            video_path=video_path,
            max_frames=max_frames,
            force_reprocess=True
        )
        igev_time = time.time() - start_time
        
        print(f"IGEV Time: {igev_time:.2f}s ({igev_time/max_frames:.3f}s/frame)")
        
    except Exception as e:
        print(f"IGEV test failed: {e}")
        igev_time = None
    
    # Test UniMatch
    print("\n--- UniMatch (New) ---")
    
    unimatch_extractor = UniMatchStereoDepthExtractor(
        batch_size=8,  # Can handle larger batches
        num_reg_refine=1,  # Start with fewer refinements for speed
        model_checkpoint="./pretrained/gmstereo-scale1-regrefine1-resumeflowthings-sceneflow-ddad84c8.pth"  # Faster model
    )
    
    start_time = time.time()
    unimatch_output = unimatch_extractor.process_video_sbs(
        video_path=video_path,
        max_frames=max_frames,
        force_reprocess=True
    )
    unimatch_time = time.time() - start_time
    
    print(f"UniMatch Time: {unimatch_time:.2f}s ({unimatch_time/max_frames:.3f}s/frame)")
    
    if igev_time:
        speedup = igev_time / unimatch_time
        print(f"\nSpeedup: {speedup:.2f}x faster!")
    
    return unimatch_output


def run_basic_pipeline(video_path: str, max_frames: int = None):
    """ Run basic UniMatch stereo depth extraction """
    print("=== BASIC UNIMATCH PIPELINE ===")
    
    # Initialize UniMatch extractor with good balance of speed/quality
    extractor = UniMatchStereoDepthExtractor(
        model_checkpoint="./pretrained/gmstereo-scale2-regrefine3-resumeflowthings-middleburyfthighres-a82bec03.pth",
        batch_size=16,
        num_reg_refine=3,
        device="cuda"
    )
    
    # Process video
    output_path = extractor.process_video_sbs(
        video_path=video_path,
        max_frames=max_frames
    )
    
    print(f"✓ Basic pipeline complete: {output_path}")
    return output_path


def run_enhanced_pipeline(video_path: str, max_frames: int = None):
    """ Run enhanced pipeline with temporal consistency """
    print("=== ENHANCED PIPELINE WITH TEMPORAL CONSISTENCY ===")
    
    # Initialize stereo extractor
    stereo_extractor = UniMatchStereoDepthExtractor(
        model_checkpoint="./pretrained/gmstereo-scale2-regrefine3-resumeflowthings-middleburyfthighres-a82bec03.pth",
        batch_size=12,  # Slightly smaller batch to leave room for flow model
        num_reg_refine=3
    )
    
    # Initialize flow processor
    flow_processor = UniMatchFlowProcessor(
        flow_model_checkpoint="./pretrained/gmflow-scale2-regrefine6-resumeflowthings-sintelft-things-1f8ac21.pth",
        num_reg_refine=6
    )
    
    # Create enhanced pipeline
    enhancer = TemporalDepthEnhancer(
        stereo_extractor=stereo_extractor,
        flow_processor=flow_processor,
        temporal_alpha=0.7,  # How much to blend with previous frame
        scene_change_threshold=0.3  # Threshold for detecting scene changes
    )
    
    # Process with temporal enhancement
    output_path = enhancer.process_video_with_temporal_consistency(
        video_path=video_path,
        max_frames=max_frames
    )
    
    print(f"✓ Enhanced pipeline complete: {output_path}")
    return output_path


def run_speed_optimized_pipeline(video_path: str, max_frames: int = None):
    """ Run speed-optimized pipeline for maximum throughput """
    print("=== SPEED-OPTIMIZED PIPELINE ===")
    
    # Use fastest model with minimal refinement
    extractor = UniMatchStereoDepthExtractor(
        model_checkpoint="./pretrained/gmstereo-scale1-regrefine1-resumeflowthings-sceneflow-ddad84c8.pth",
        batch_size=32,  # Large batch size
        num_scales=1,   # Single scale
        num_reg_refine=1,  # Minimal refinement
        inference_size=[384, 512],  # Smaller inference size for speed
        device="cuda"
    )
    
    # Process video
    output_path = extractor.process_video_sbs(
        video_path=video_path,
        max_frames=max_frames
    )
    
    print(f"✓ Speed-optimized pipeline complete: {output_path}")
    return output_path


def run_quality_optimized_pipeline(video_path: str, max_frames: int = None):
    """ Run quality-optimized pipeline for best results """
    print("=== QUALITY-OPTIMIZED PIPELINE ===")
    
    # Use best model with maximum refinement
    extractor = UniMatchStereoDepthExtractor(
        model_checkpoint="./pretrained/gmstereo-scale2-regrefine3-resumeflowthings-middleburyfthighres-a82bec03.pth",
        batch_size=4,   # Smaller batch for memory
        num_scales=2,   # Multi-scale
        num_reg_refine=6,  # Maximum refinement
        inference_size=None,  # Full resolution
        device="cuda"
    )
    
    # Process video
    output_path = extractor.process_video_sbs(
        video_path=video_path,
        max_frames=max_frames
    )
    
    print(f"✓ Quality-optimized pipeline complete: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description='UniMatch integration example for video 3D pipeline')
    parser.add_argument('video', help='Path to SBS video file')
    parser.add_argument('--mode', 
                       choices=['benchmark', 'basic', 'enhanced', 'speed', 'quality'], 
                       default='basic',
                       help='Pipeline mode to run')
    parser.add_argument('--max-frames', type=int, default=None,
                       help='Maximum frames to process (default: all)')
    parser.add_argument('--benchmark-frames', type=int, default=50,
                       help='Frames to use for benchmarking (default: 50)')
    
    args = parser.parse_args()
    
    video_path = args.video
    if not Path(video_path).exists():
        print(f"Error: Video file not found: {video_path}")
        sys.exit(1)
    
    print(f"Processing video: {video_path}")
    print(f"Mode: {args.mode}")
    
    try:
        if args.mode == 'benchmark':
            benchmark_comparison(video_path, args.benchmark_frames)
        elif args.mode == 'basic':
            run_basic_pipeline(video_path, args.max_frames)
        elif args.mode == 'enhanced':
            run_enhanced_pipeline(video_path, args.max_frames)
        elif args.mode == 'speed':
            run_speed_optimized_pipeline(video_path, args.max_frames)
        elif args.mode == 'quality':
            run_quality_optimized_pipeline(video_path, args.max_frames)
        
        print("\n✓ Processing complete!")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
