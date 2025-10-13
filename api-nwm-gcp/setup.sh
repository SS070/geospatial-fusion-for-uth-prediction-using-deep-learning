#!/bin/bash

# Complete Setup Script for UTH Prediction API
# This script sets up the entire project structure and dependencies

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}$1${NC}"
}

# Print header
print_header "=============================================="
print_header "  UTH Prediction API - Complete Setup"
print_header "=============================================="

# Check system requirements
print_status "Checking system requirements..."

# Check Python version
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is not installed. Please install Python 3.9+ and try again."
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1-2)
REQUIRED_VERSION="3.9"

if ! python3 -c "import sys; exit(0 if sys.version_info >= (3,9) else 1)"; then
    print_error "Python 3.9+ required. Found: Python $PYTHON_VERSION"
    exit 1
fi

print_status "✅ Python $PYTHON_VERSION detected"

# Check for GPU support
if command -v nvidia-smi &> /dev/null; then
    print_status "✅ NVIDIA GPU detected"
    GPU_AVAILABLE=true
else
    print_warning "⚠️  No NVIDIA GPU detected. CPU-only mode will be used."
    GPU_AVAILABLE=false
fi

# Check Docker
if command -v docker &> /dev/null; then
    print_status "✅ Docker detected"
    DOCKER_AVAILABLE=true
else
    print_warning "⚠️  Docker not found. Docker features will be unavailable."
    DOCKER_AVAILABLE=false
fi

# Create project structure
print_status "Creating project structure..."

# Main directories
mkdir -p with_saphir
mkdir -p without_saphir
mkdir -p additional_models/{geo_transformer,geo_mean_fusion,geo_dynamic_2018,geo_bilinear_interpolation}
mkdir -p data/{MICROWAVE_UTH_DATA_NOAA,SAPHIR_RH_DATA_PROCESSED_2}
mkdir -p outputs/{with_saphir,without_saphir}
mkdir -p checkpoints/{with_saphir,without_saphir}
mkdir -p logs
mkdir -p tests

print_status "✅ Project directories created"

# Create virtual environment
print_status "Setting up Python virtual environment..."

if [ ! -d "venv" ]; then
    python3 -m venv venv
    print_status "✅ Virtual environment created"
else
    print_warning "Virtual environment already exists"
fi

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
print_status "Upgrading pip..."
pip install --upgrade pip

# Install requirements
print_status "Installing Python dependencies..."
pip install -r requirements.txt

print_status "✅ Dependencies installed"

# Move files to correct locations
print_status "Organizing files..."

# Move config files
if [ -f "with_saphir_config.py" ]; then
    mv with_saphir_config.py with_saphir/config.py
    print_status " SAPHIR config moved to with_saphir/config.py"
fi

if [ -f "without_saphir_config.py" ]; then
    mv without_saphir_config.py without_saphir/config.py
    print_status " Non-SAPHIR config moved to without_saphir/config.py"
fi

# Move training scripts
if [ -f "with_saphir_train.py" ]; then
    mv with_saphir_train.py with_saphir/<select file>.py
    print_status " SAPHIR training script moved to with_saphir/<select file>.py"
fi

if [ -f "without_saphir_train.py" ]; then
    mv without_saphir_train.py without_saphir/<select file>.py
    print_status " Non-SAPHIR training script moved to without_saphir/<select file>.py"
fi

# Move inference script
if [ -f "with_saphir_inference.py" ]; then
    mv with_saphir_inference.py with_saphir/inference.py
    print_status " SAPHIR inference script moved to with_saphir/inference.py"
fi

# Create __init__.py files
touch with_saphir/__init__.py
touch without_saphir/__init__.py
touch additional_models/__init__.py

# Create quick start scripts
print_status "Creating quick start scripts..."

# Create training launcher
cat > train_with_saphir.sh << 'EOF'
#!/bin/bash
echo "🚀 Starting training with SAPHIR integration..."
source venv/bin/activate
cd with_saphir
python <select file>.py "$@"
EOF

cat > train_without_saphir.sh << 'EOF'
#!/bin/bash
echo "🚀 Starting training without SAPHIR (global coverage)..."
source venv/bin/activate
cd without_saphir
python <select file>.py "$@"
EOF

# Create inference launcher
cat > run_inference_saphir.sh << 'EOF'
#!/bin/bash
echo "🔮 Running inference with SAPHIR model..."
source venv/bin/activate
cd with_saphir
python inference.py "$@"
EOF

# Make scripts executable
chmod +x train_with_saphir.sh
chmod +x train_without_saphir.sh  
chmod +x run_inference_saphir.sh
chmod +x build_docker.sh

print_status "✅ Quick start scripts created"

# Create test data structure info
cat > data/README.md << 'EOF'
# Data Directory Structure

This directory should contain your satellite data. Please organize it as follows:

## Microwave Data (Required)
```
MICROWAVE_UTH_DATA_NOAA/
├── AMSU_B_NOAA_15_1999_2002/
├── AMSU_B_NOAA_16_2001_2006/
├── AMSU_B_NOAA_17_2003_2008/
├── MHS_NOAA_18_2006_2018/
├── MHS_NOAA_19_2016_2021/
├── MHS_Metop_A_2007_2021/
├── MHS_Metop_B_2013_2021/
└── MHS_MetOp_C_2019_2021/
```

Each subdirectory should contain NetCDF (.nc) files with the UTH data.

## SAPHIR Data (Optional - for enhanced accuracy)
```
SAPHIR_RH_DATA_PROCESSED_2/
├── 2012/
├── 2013/
├── ...
└── 2021/
```

Each year directory should contain SAPHIR humidity data files.

## File Naming Convention
- Microwave files: Must contain date in YYYYMMDD format (position 5-13 in filename)
- SAPHIR files: uthsaphirrh[YYYYMMDD].nc format

## Permissions
Ensure the data directory has proper read permissions:
```bash
chmod -R 755 data/
```

## Troubleshooting
If you encounter data loading issues:
1. Check file permissions
2. Verify NetCDF file integrity with `ncdump -h filename.nc`
3. Ensure consistent coordinate systems across files
EOF

# Create environment configuration
cat > .env.example << 'EOF'
# Environment Configuration for UTH Prediction API
# Copy to .env and modify paths according to your setup

# Data Directories (modify these paths)
MICROWAVE_DATA_DIR=/path/to/your/MICROWAVE_UTH_DATA_NOAA
SAPHIR_DATA_DIR=/path/to/your/SAPHIR_RH_DATA_PROCESSED_2

# Output Directories
CHECKPOINT_DIR=./checkpoints
OUTPUT_DIR=./outputs

# GPU Settings
CUDA_VISIBLE_DEVICES=0

# Training Settings
EPOCHS=25
LEARNING_RATE=1e-4
BATCH_SIZE=1

# Docker Settings
DOCKER_IMAGE_TAG=uth-prediction:latest
EOF

# Test basic functionality
print_status "Testing basic functionality..."

# Test imports
python3 -c "
import torch
import numpy as np
import xarray as xr
print('✅ Core dependencies imported successfully')
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA device: {torch.cuda.get_device_name()}')
"

# Create basic test
cat > test_setup.py << 'EOF'
#!/usr/bin/env python3
"""
Basic functionality test for UTH Prediction API setup
"""
import sys
import os
import torch
import numpy as np
import xarray as xr
from datetime import datetime

def test_imports():
    """Test that all required packages can be imported"""
    try:
        import torch
        import torchvision
        import numpy as np
        import xarray as xr
        import matplotlib.pyplot as plt
        import sklearn
        import tqdm
        print("✅ All required packages imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_pytorch():
    """Test PyTorch functionality"""
    try:
        # Test basic operations
        x = torch.randn(2, 3)
        y = torch.randn(3, 2)
        z = torch.mm(x, y)
        
        # Test CUDA if available
        if torch.cuda.is_available():
            x_gpu = x.cuda()
            print(f"✅ PyTorch with CUDA working (Device: {torch.cuda.get_device_name()})")
        else:
            print("✅ PyTorch working (CPU only)")
        
        return True
    except Exception as e:
        print(f"❌ PyTorch error: {e}")
        return False

def test_data_directories():
    """Test that data directories exist"""
    dirs_to_check = [
        'data',
        'with_saphir',
        'without_saphir',
        'outputs',
        'checkpoints'
    ]
    
    all_exist = True
    for dir_name in dirs_to_check:
        if os.path.exists(dir_name):
            print(f"✅ Directory {dir_name} exists")
        else:
            print(f"❌ Directory {dir_name} missing")
            all_exist = False
    
    return all_exist

def main():
    print("🧪 Running setup verification tests...")
    print("=" * 50)
    
    results = []
    results.append(test_imports())
    results.append(test_pytorch())
    results.append(test_data_directories())
    
    print("=" * 50)
    if all(results):
        print("🎉 All tests passed! Setup is complete.")
        return 0
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
EOF

chmod +x test_setup.py

# Run the test
python3 test_setup.py

# Docker setup
if [ "$DOCKER_AVAILABLE" = true ]; then
    print_status "Setting up Docker..."
    
    # Test Docker build (optional)
    read -p "Would you like to build the Docker image now? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_status "Building Docker image..."
        ./build_docker.sh
    else
        print_status "Docker image build skipped. Run './build_docker.sh' later to build."
    fi
fi

# Create final summary
print_header "=============================================="
print_header "           Setup Complete! 🎉"
print_header "=============================================="

print_status "Project structure created successfully!"
print_status ""
print_status "📁 Directory Structure:"
print_status "├── with_saphir/          # SAPHIR-enhanced implementation"
print_status "├── without_saphir/       # Global coverage implementation" 
print_status "├── additional_models/    # Alternative model implementations"
print_status "├── data/                 # Your satellite data (see data/README.md)"
print_status "├── outputs/              # Model outputs and predictions"
print_status "├── checkpoints/          # Trained model checkpoints"
print_status "└── logs/                 # Training and inference logs"
print_status ""

print_status "🚀 Quick Start Commands:"
print_status ""
print_status "1. Configure your data paths:"
print_status "   - Edit with_saphir/config.py"
print_status "   - Edit without_saphir/config.py"
print_status "   - Or set environment variables (see .env.example)"
print_status ""
print_status "2. Train models:"
print_status "   ./train_with_saphir.sh      # Enhanced accuracy with SAPHIR"
print_status "   ./train_without_saphir.sh   # Global coverage, microwave only"
print_status ""
print_status "3. Run inference:"
print_status "   ./run_inference_saphir.sh --checkpoint path/to/model.pth"
print_status ""
print_status "4. Docker deployment:"
if [ "$DOCKER_AVAILABLE" = true ]; then
    print_status "   ./build_docker.sh           # Build Docker image"
    print_status "   docker run -it --gpus all uth-prediction:latest"
else
    print_warning "   Docker not available - install Docker for containerized deployment"
fi

print_status ""
print_status "📚 Documentation:"
print_status "   - README.md          # Main documentation"
print_status "   - DEVELOPMENT.md     # Developer guide"
print_status "   - data/README.md     # Data structure guide"
print_status ""

print_status "💡 Next Steps:"
print_status "1. Place your satellite data in the data/ directory"
print_status "2. Update configuration files with correct data paths"
print_status "3. Run training: ./train_with_saphir.sh or ./train_without_saphir.sh"
print_status "4. Check README.md for detailed instructions"

print_status ""
print_status " Need Help?"
print_status "   - Email: saishashank3000@gmail.com, kvsm2k@gmail.com"
print_status "   - GitHub Issues: https://github.com/BYU-Hydroinformatics/api-nwm-gcp/issues"

print_header "=============================================="

# Deactivate virtual environment
deactivate

print_status "Setup completed successfully! "
print_status "Remember to activate the virtual environment before running: source venv/bin/activate"