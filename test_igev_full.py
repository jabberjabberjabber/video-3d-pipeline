#!/usr/bin/env python3
"""Test IGEV with a model that bypasses the problematic Feature class."""

import sys
from pathlib import Path

# Add IGEV to path
project_root = Path(__file__).parent
igev_path = project_root / "IGEV"
sys.path.append(str(igev_path))
sys.path.append(str(igev_path / "core"))

print(f"Testing IGEV with model loading...")

try:
    import torch
    from core.igev_stereo import IGEVStereo
    from core.utils.utils import InputPadder
    
    print("✓ Imports successful")
    
    # Find available model
    models_dir = project_root / "models" / "Selective-IGEV"
    model_file = None
    
    for subdir in ["sceneflow", "eth3d", "kitti", "middlebury"]:
        model_path = models_dir / subdir
        if model_path.exists():
            pth_files = list(model_path.glob("*.pth"))
            if pth_files:
                model_file = pth_files[0]
                print(f"Found model: {model_file}")
                break
    
    if not model_file:
        print("❌ No .pth model files found in models/Selective-IGEV/")
        sys.exit(1)
    
    # Test loading the actual model
    print("Testing model loading...")
    
    class Args:
        def __init__(self):
            self.mixed_precision = True
            self.precision_dtype = "float16"
            self.hidden_dims = [128] * 3
            self.corr_implementation = "reg"
            self.shared_backbone = False
            self.corr_levels = 2
            self.corr_radius = 4
            self.n_downsample = 2
            self.slow_fast_gru = False
            self.n_gru_layers = 3
            self.max_disp = 192
    
    args = Args()
    
    # Create model
    model = torch.nn.DataParallel(IGEVStereo(args), device_ids=[0])
    
    # Load checkpoint
    print(f"Loading checkpoint: {model_file}")
    checkpoint = torch.load(str(model_file), map_location='cpu')
    model.load_state_dict(checkpoint)
    
    model = model.module
    model.eval()
    
    print("✓ Model loaded successfully!")
    
    # Test with dummy input
    print("Testing inference...")
    with torch.no_grad():
        dummy_left = torch.randn(1, 3, 256, 512)
        dummy_right = torch.randn(1, 3, 256, 512)
        
        padder = InputPadder(dummy_left.shape, divis_by=32)
        left_padded, right_padded = padder.pad(dummy_left, dummy_right)
        
        # Test inference
        result = model(left_padded, right_padded, iters=12, test_mode=True)
        result = padder.unpad(result)
        
        print(f"✓ Inference successful! Output shape: {result.shape}")
    
    print("\n🎉 IGEV is working! Your depth.py should work now.")
    print(f"Use this model: {model_file}")
        
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
