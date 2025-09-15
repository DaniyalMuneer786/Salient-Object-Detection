#!/usr/bin/env python3
"""
Test script to debug AI model loading issues
"""

import os
import sys
import traceback

def test_imports():
    """Test if all required imports are available"""
    print("Testing imports...")
    
    try:
        import torch
        print(f"✓ PyTorch version: {torch.__version__}")
    except ImportError as e:
        print(f"✗ PyTorch import failed: {e}")
        return False
    
    try:
        import torchvision
        print(f"✓ Torchvision version: {torchvision.__version__}")
    except ImportError as e:
        print(f"✗ Torchvision import failed: {e}")
        return False
    
    try:
        import yaml
        print("✓ PyYAML imported successfully")
    except ImportError as e:
        print(f"✗ PyYAML import failed: {e}")
        return False
    
    try:
        from utils.misc import get_model
        print("✓ get_model function imported successfully")
    except ImportError as e:
        print(f"✗ get_model import failed: {e}")
        return False
    
    try:
        from base_structure import BaseStructure
        print("✓ BaseStructure class imported successfully")
    except ImportError as e:
        print(f"✗ BaseStructure import failed: {e}")
        return False
    
    return True

def test_files():
    """Test if model files exist"""
    print("\nTesting model files...")
    
    MODEL_PATH = "ckpt/nq20_ndl6_bc_sr10100_duts_pm_seed0_contrastive/latest_model.pt"
    CONFIG_PATH = "configs/duts-dino-k234-nq20-224-swav-mocov2-dino-p16-sr10100.yaml"
    
    model_exists = os.path.exists(MODEL_PATH)
    config_exists = os.path.exists(CONFIG_PATH)
    
    print(f"Model file exists: {model_exists}")
    if model_exists:
        print(f"  Path: {MODEL_PATH}")
        print(f"  Size: {os.path.getsize(MODEL_PATH) / (1024*1024):.2f} MB")
    else:
        print(f"  Missing: {MODEL_PATH}")
    
    print(f"Config file exists: {config_exists}")
    if config_exists:
        print(f"  Path: {CONFIG_PATH}")
    else:
        print(f"  Missing: {CONFIG_PATH}")
    
    return model_exists and config_exists

def test_model_loading():
    """Test if the model can be loaded"""
    print("\nTesting model loading...")
    
    try:
        import torch
        import yaml
        from argparse import Namespace
        from utils.misc import get_model
        from base_structure import BaseStructure
        
        MODEL_PATH = "ckpt/nq20_ndl6_bc_sr10100_duts_pm_seed0_contrastive/latest_model.pt"
        CONFIG_PATH = "configs/duts-dino-k234-nq20-224-swav-mocov2-dino-p16-sr10100.yaml"
        
        # Load configuration
        print("Loading configuration...")
        with open(CONFIG_PATH, 'r') as f:
            config = yaml.safe_load(f)
        if isinstance(config, dict):
            config = Namespace(**config)
        print("✓ Configuration loaded")
        
        # Setup device
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"✓ Using device: {device}")
        
        # Initialize model
        print("Initializing model...")
        model = get_model(
            arch="maskformer",
            configs=config
        ).to(device)
        print("✓ Model initialized")
        
        # Load weights
        print("Loading model weights...")
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(checkpoint['model'])
        model.eval()
        print("✓ Model weights loaded")
        
        # Initialize base structure
        print("Initializing base structure...")
        base_structure = BaseStructure(
            model=model,
            device=device
        )
        print("✓ Base structure initialized")
        
        print("✓ Model loading test completed successfully!")
        return True
        
    except Exception as e:
        print(f"✗ Model loading failed: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        print("Traceback:")
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("=== AI Model Loading Test ===\n")
    
    # Test imports
    imports_ok = test_imports()
    
    # Test files
    files_ok = test_files()
    
    # Test model loading
    model_ok = False
    if imports_ok and files_ok:
        model_ok = test_model_loading()
    
    # Summary
    print("\n=== Test Summary ===")
    print(f"Imports: {'✓ PASS' if imports_ok else '✗ FAIL'}")
    print(f"Files: {'✓ PASS' if files_ok else '✗ FAIL'}")
    print(f"Model Loading: {'✓ PASS' if model_ok else '✗ FAIL'}")
    
    if imports_ok and files_ok and model_ok:
        print("\n🎉 All tests passed! The AI model should work correctly.")
    else:
        print("\n❌ Some tests failed. Please check the errors above.")
        if not imports_ok:
            print("  - Install missing dependencies: pip install torch torchvision pyyaml")
        if not files_ok:
            print("  - Ensure model files are in the correct locations")
        if not model_ok:
            print("  - Check model configuration and weights")

if __name__ == "__main__":
    main() 