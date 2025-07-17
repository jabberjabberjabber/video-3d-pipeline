"""
Optimized UniMatch configurations for different hardware and use cases.
"""

from .unimatch_depth import UniMatchStereoDepthExtractor


def rtx_3080_speed_config(model_path: str = None) -> dict:
    """Ultra-fast configuration for RTX 3080 - targets <100ms per frame"""
    
    if model_path is None:
        # Use fastest available model
        model_path = "./models/gmstereo-scale1-regrefine1-resumeflowthings-sceneflow-ddad84c8.pth"
    
    return {
        'model_checkpoint': model_path,
        'batch_size': 64,  # Very aggressive batching
        'num_scales': 1,
        'num_reg_refine': 1,
        'inference_size': [384, 640],  # Small inference size
        'feature_channels': 64,  # Reduced channels
        'attn_type': 'self_swin2d_cross_1d',
        'attn_splits_list': [2],
        'corr_radius_list': [-1],
        'prop_radius_list': [-1],
        'padding_factor': 32
    }


def rtx_3080_balanced_config(model_path: str = None) -> dict:
    """Balanced speed/quality for RTX 3080"""
    
    if model_path is None:
        model_path = "./models/gmstereo-scale2-regrefine3-resumeflowthings-middleburyfthighres-a82bec03.pth"
    
    return {
        'model_checkpoint': model_path,
        'batch_size': 32,
        'num_scales': 1,
        'num_reg_refine': 2,
        'inference_size': [480, 854],  # ~50% of 1080p
        'attn_type': 'self_swin2d_cross_1d',
        'attn_splits_list': [2],
        'corr_radius_list': [-1],
        'prop_radius_list': [-1]
    }


def rtx_3080_quality_config(model_path: str = None) -> dict:
    """Best quality while still fast on RTX 3080"""
    
    if model_path is None:
        model_path = "./models/gmstereo-scale2-regrefine3-resumeflowthings-middleburyfthighres-a82bec03.pth"
    
    return {
        'model_checkpoint': model_path,
        'batch_size': 16,
        'num_scales': 2,
        'num_reg_refine': 3,
        'inference_size': [576, 1024],  # Higher resolution
        'attn_type': 'self_swin2d_cross_swin1d',
        'attn_splits_list': [2, 8],
        'corr_radius_list': [-1, 4],
        'prop_radius_list': [-1, 1]
    }


def create_extractor_for_rtx_3080(mode: str = 'speed', **kwargs) -> UniMatchStereoDepthExtractor:
    """
    Create optimized extractor for RTX 3080
    
    Args:
        mode: 'speed', 'balanced', or 'quality'
        **kwargs: Override any configuration parameters
    """
    
    if mode == 'speed':
        config = rtx_3080_speed_config()
    elif mode == 'balanced':
        config = rtx_3080_balanced_config()
    elif mode == 'quality':
        config = rtx_3080_quality_config()
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'speed', 'balanced', or 'quality'")
    
    # Apply any overrides
    config.update(kwargs)
    
    print(f"Creating RTX 3080 {mode} extractor:")
    print(f"  Batch size: {config['batch_size']}")
    print(f"  Inference size: {config.get('inference_size', 'Full resolution')}")
    print(f"  Refinement iterations: {config['num_reg_refine']}")
    print(f"  Expected speed: {_get_expected_speed(mode)}")
    
    return UniMatchStereoDepthExtractor(**config)


def _get_expected_speed(mode: str) -> str:
    """Get expected processing speed for each mode"""
    speeds = {
        'speed': '50-100ms per frame (~10-20 FPS)',
        'balanced': '100-200ms per frame (~5-10 FPS)', 
        'quality': '200-500ms per frame (~2-5 FPS)'
    }
    return speeds.get(mode, 'Unknown')


def benchmark_configurations(video_path: str, max_frames: int = 10):
    """
    Benchmark different configurations to find optimal settings
    """
    import time
    
    configurations = [
        ('Ultra Speed', 'speed'),
        ('Balanced', 'balanced'),
        ('Quality', 'quality')
    ]
    
    results = {}
    
    for name, mode in configurations:
        print(f"\n=== Testing {name} Configuration ===")
        
        try:
            extractor = create_extractor_for_rtx_3080(mode)
            
            start_time = time.time()
            output_path = extractor.process_video_sbs(
                video_path=video_path,
                max_frames=max_frames,
                force_reprocess=True
            )
            end_time = time.time()
            
            total_time = end_time - start_time
            time_per_frame = total_time / max_frames
            fps = 1 / time_per_frame
            
            results[name] = {
                'total_time': total_time,
                'time_per_frame': time_per_frame,
                'fps': fps,
                'output_path': output_path
            }
            
            print(f"✓ {name}: {time_per_frame*1000:.1f}ms/frame ({fps:.1f} FPS)")
            
        except Exception as e:
            print(f"✗ {name} failed: {e}")
            results[name] = {'error': str(e)}
    
    # Print summary
    print(f"\n=== Benchmark Summary ===")
    for name, result in results.items():
        if 'error' not in result:
            print(f"{name:12}: {result['time_per_frame']*1000:6.1f}ms/frame ({result['fps']:4.1f} FPS)")
        else:
            print(f"{name:12}: FAILED - {result['error']}")
    
    return results


# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test UniMatch configurations')
    parser.add_argument('video', help='Path to test video')
    parser.add_argument('--benchmark', action='store_true', 
                       help='Run benchmark of all configurations')
    parser.add_argument('--mode', choices=['speed', 'balanced', 'quality'], 
                       default='speed', help='Configuration mode')
    parser.add_argument('--max-frames', type=int, default=10,
                       help='Frames for testing (default: 10)')
    
    args = parser.parse_args()
    
    if args.benchmark:
        benchmark_configurations(args.video, args.max_frames)
    else:
        extractor = create_extractor_for_rtx_3080(args.mode)
        output_path = extractor.process_video_sbs(
            video_path=args.video,
            max_frames=args.max_frames
        )
        print(f"Output: {output_path}")
