#!/usr/bin/env python3
"""Test IGEV imports to identify the exact issue."""

import sys
from pathlib import Path

# Add IGEV to path
project_root = Path(__file__).parent
igev_path = project_root / "IGEV"
sys.path.append(str(igev_path))
sys.path.append(str(igev_path / "core"))

print(f"Testing IGEV imports...")
print(f"Project root: {project_root}")
print(f"IGEV path: {igev_path}")
print(f"IGEV exists: {igev_path.exists()}")

try:
    print("Importing torch...")
    import torch
    print(f"✓ torch version: {torch.__version__}")
    
    print("Importing core.igev_stereo...")
    from core.igev_stereo import IGEVStereo
    print("✓ IGEVStereo imported")
    
    print("Importing core.utils.utils...")
    from core.utils.utils import InputPadder
    print("✓ InputPadder imported")
    
    print("Testing model creation...")
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
    model = IGEVStereo(args)
    print("✓ IGEV model created successfully")
    
    print("\n🎉 All imports successful! Your depth.py should work.")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("\nMissing dependencies. Install with:")
    print("pip install torch torchvision")
    
except Exception as e:
    print(f"❌ Other error: {e}")
    print("Check IGEV directory structure")
