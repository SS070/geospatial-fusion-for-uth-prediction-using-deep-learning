# Geospatial UTH Prediction API - Complete Implementation

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9+-green.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Docker](https://img.shields.io/badge/docker-available-blue.svg)](https://hub.docker.com/)

> **Contact**: saishashank3000@gmail.com, kvsm2k@gmail.com  
> **Date First Available**: October 13, 2025  
> **Software Required**: Docker (all other software is installed within Docker image)  
> **Program Language**: Python  

Deep learning models for atmospheric humidity prediction using multi-satellite fusion with UNet++ VAE architecture. Supports both SAPHIR-enhanced (±30° latitude priority) and global coverage implementations.

##  Features

- **Dual Implementation**: Choose between SAPHIR-enhanced accuracy or global coverage
- **Advanced Architecture**: UNet++ with Variational Autoencoder (VAE) bottleneck
- **Multi-Satellite Fusion**: Quality-weighted fusion of 7-8 satellite sources
- **Docker Ready**: Complete containerization for easy deployment
- **Production Ready**: Comprehensive logging, monitoring, and error handling
- **Scalable**: Optimized for both research and operational use

##  Quick Start

### Option 1: Automated Setup (Recommended)
```bash
# Clone repository
git clone https://github.com/BYU-Hydroinformatics/api-nwm-gcp.git
cd api-nwm-gcp

# Run complete setup
chmod +x setup.sh
./setup.sh
```

### Option 2: Docker-Only Setup
```bash
# Clone and build Docker image
git clone https://github.com/BYU-Hydroinformatics/api-nwm-gcp.git
cd api-nwm-gcp

chmod +x build_docker.sh
./build_docker.sh

# Run with your data
docker run -it --gpus all \
  -v /path/to/your/data:/data \
  -v /path/to/your/output:/output \
  uth-prediction:latest
```

##  Project Structure

```
api-nwm-gcp/
├──  with_saphir/              # Enhanced accuracy implementation
│   ├── config.py                # SAPHIR-specific configuration
│   ├── Geo_Spatial_Fusion_UNET++_VAE_KL_NLL.py   # Training Script 1 
│   ├── GeoSpatial_Fusion_UNET++_VAE.py           # Training script 2
│   └── inference.py             # Inference with SAPHIR models
├──  without_saphir/        # Alternative implementations
│   ├── geo_transformer.py       # Transformer-based model
│   ├── geo_mean_fusion.py       # Simple fusion baseline
│   ├── geo_dynamic_2018.py      # Dynamic fusion approach
│   └── geo_bilinear_interpolation.py  # Interpolation baseline
├──  data/                     # Your satellite data directory
│   ├── MICROWAVE_UTH_DATA_NOAA/ # Required: microwave satellite data
│   └── SAPHIR_RH_DATA_PROCESSED_2/ # Optional: SAPHIR data
├──  outputs/                  # Model predictions and results
├──  checkpoints/             # Trained model weights
├──  Dockerfile               # Container definition
├──  requirements.txt          # Python dependencies
├──  setup.sh                 # Automated setup script
├──  README.md                # This file
└──  DEVELOPMENT.md           # Developer documentation
```

##  Data Requirements

### Directory Structure
Your data should be organized as follows:

```
data/
├── MICROWAVE_UTH_DATA_NOAA/          #  REQUIRED
│   ├── AMSU_B_NOAA_15_1999_2002/
│   ├── AMSU_B_NOAA_16_2001_2006/
│   ├── AMSU_B_NOAA_17_2003_2008/
│   ├── MHS_NOAA_18_2006_2018/
│   ├── MHS_NOAA_19_2016_2021/
│   ├── MHS_Metop_A_2007_2021/
│   ├── MHS_Metop_B_2013_2021/
│   └── MHS_MetOp_C_2019_2021/
└── SAPHIR_RH_DATA_PROCESSED_2/       # OPTIONAL (for enhanced accuracy)
    ├── 2012/
    │   ├── uthsaphirrh20120101.nc
    │   ├── uthsaphirrh20120102.nc
    │   └── ...
    ├── 2013/
    └── ...
```

### File Format Requirements
- **Format**: NetCDF (.nc) files
- **Microwave Data**: Must contain `uth_mean_ascend_descend` variable
- **SAPHIR Data**: Must contain `rh`, `lat`, `lon` variables  
- **Naming Convention**: Date in YYYYMMDD format (position 5-13 in filename)
- **Grid**: 180×360 global grid (1° resolution)
- **Coordinates**: Standard latitude (-89.5 to 89.5) and longitude (-179.5 to 179.5)

##  Usage

### Training Models

#### With SAPHIR (Enhanced Accuracy in ±30° Latitude Band)
```bash
# Activate environment
source venv/bin/activate

# Quick start
./train_with_saphir.sh

# Or manually:
cd with_saphir/
python <select file>.py
```

**Features:**
-  Enhanced accuracy in tropical/subtropical regions (±30°)
- Superior data quality from SAPHIR sensor
-  Reduced uncertainty in critical climate zones
-  Limited to SAPHIR availability period (2012-2021)

#### Without SAPHIR (Global Coverage)
```bash
# Activate environment  
source venv/bin/activate

# Quick start
./train_without_saphir.sh

# Or manually:
cd without_saphir/
python <select file>.py
```

**Features:**
-  Complete global coverage
-  Extended temporal range (1999-2021)
-  Consistent data availability
- ️ Lower accuracy compared to SAPHIR-enhanced regions

### Running Inference

#### Single Date Prediction
```bash
# SAPHIR model
./run_inference_saphir.sh \
    --checkpoint ./checkpoints/with_saphir/best_model.pth \
    --single-date 2020-06-15 \
    --output-dir ./outputs

```

#### Batch Processing
```bash
# Process multiple dates (e.g., first 100 samples)
python with_saphir/inference.py \
    --checkpoint ./checkpoints/with_saphir/best_model.pth \
    --max-samples 100 \
    --output-dir ./outputs
```

### Docker Deployment

#### Building the Container
```bash
# Build with default settings
./build_docker.sh

# Build with custom tag
./build_docker.sh --tag my-uth-model:v1.0

# Build without cache (fresh build)
./build_docker.sh --no-cache
```

#### Running the Container
```bash
# With GPU support (recommended)
docker run -it --gpus all \
  -v /path/to/your/data:/data \
  -v /path/to/your/output:/output \
  uth-prediction:latest bash

# CPU-only mode
docker run -it \
  -v /path/to/your/data:/data \
  -v /path/to/your/output:/output \
  uth-prediction:latest bash

# Inside container, run:
cd with_saphir/
python <select file>.py  # or inference.py
```

## ⚙️ Configuration

### Environment Variables
```bash
# Set in your shell or .env file
export MICROWAVE_DATA_DIR="/path/to/MICROWAVE_UTH_DATA_NOAA"
export SAPHIR_DATA_DIR="/path/to/SAPHIR_RH_DATA_PROCESSED_2"
export CHECKPOINT_DIR="./checkpoints"
export OUTPUT_DIR="./outputs"
export CUDA_VISIBLE_DEVICES="0"
```

### Configuration Files
Modify the configuration files to match your setup:

**with_saphir/config.py:**
```python
# Update these paths
BASE_DIR = "/your/path/to/MICROWAVE_UTH_DATA_NOAA"
SAPHIR_BASE_DIR = "/your/path/to/SAPHIR_RH_DATA_PROCESSED_2"

# Adjust model parameters
MODEL_CONFIG = {
    'in_channels': 1,
    'base': 64,          # Increase for more model capacity
    'out_channels': 1,
    'num_sat': 8,
    'dropout': 0.1,
    'latent_dim': 512
}

# Modify training settings
EPOCHS = 25              # Increase for longer training
LEARNING_RATE = 1e-4     # Adjust learning rate
BATCH_SIZE = 1           # Increase if you have more GPU memory
```

##  Model Performance

### Expected Performance Metrics

| Implementation | RMSE | MAE | SSIM | Coverage | Speed (samples/sec) |
|---------------|------|-----|------|----------|-------------------|
| **With SAPHIR** | 0.12-0.15 | 0.08-0.12 | 0.85-0.92 | ±30° lat | ~2.5 |
| **Without SAPHIR** | 0.15-0.20 | 0.12-0.16 | 0.80-0.88 | Global | ~3.2 |

### Training Time Estimates
- **With SAPHIR**: ~2-4 hours (25 epochs, RTX 3090)
- **Without SAPHIR**: ~3-6 hours (30 epochs, RTX 3090)
- **CPU-only**: 10-20x slower than GPU

## 🛠️ Advanced Usage

### Custom Model Architecture
```python
# Modify model configuration for different architectures
MODEL_CONFIG = {
    'in_channels': 1,
    'base': 32,          # Lighter model for faster training
    'out_channels': 1,
    'num_sat': 8,
    'dropout': 0.2,      # Higher dropout for regularization
    'latent_dim': 256    # Smaller latent space
}
```

### Hyperparameter Tuning
```python
# Loss function weights (with_saphir/config.py)
LOSS_WEIGHTS = {
    'nll': 1.0,
    'charbonnier': 0.5,
    'kl': 0.01,          # Increase for more regularization
    'gradient': 0.1      # Decrease for less smoothing
}

# Optimization settings
OPTIMIZER_CONFIG = {
    'lr': 5e-5,          # Lower learning rate for stability
    'weight_decay': 1e-5 # L2 regularization
}
```

### Multi-GPU Training
```bash
# Set multiple GPUs
export CUDA_VISIBLE_DEVICES="0,1,2,3"

# Update batch size for multi-GPU
BATCH_SIZE = 4  # in config.py
```

##  Output Files

### Training Outputs
```
checkpoints/
├── with_saphir/
│   ├── best_model.pth              # Best validation model
│   ├── UNET++_VAE_Saphir_FINAL.pth # Final model
│   ├── checkpoint_epoch_*.pth       # Periodic checkpoints
│   └── training_curves.png          # Loss visualization
└── without_saphir/
    ├── best_model.pth              # Best validation model
    ├── UNET++_VAE_NoSaphir_FINAL.pth # Final model
    └── checkpoint_epoch_*.pth       # Periodic checkpoints
```

### Inference Outputs
```
outputs/
├── uth_predictions_final_*.nc      # NetCDF with predictions
├── uth_predictions_intermediate_*.nc # Intermediate results
├── inference_statistics.txt        # Performance stats
└── sample_predictions.png          # Visualization
```

### NetCDF Output Structure
```python
# Variables in output NetCDF files
{
    'uth_prediction': (['time', 'lat', 'lon'], pred_array),
    'uth_uncertainty': (['time', 'lat', 'lon'], uncertainty_array),
}

# Coordinates
{
    'time': datetime_array,     # Sample dates
    'lat': [-89.5, ..., 89.5],  # Latitude grid
    'lon': [-179.5, ..., 179.5] # Longitude grid
}
```

##  Troubleshooting

### Common Issues and Solutions

#### 1. CUDA Out of Memory
```bash
# Reduce batch size
BATCH_SIZE = 1

# Clear GPU cache in training loop
torch.cuda.empty_cache()

# Use gradient checkpointing
USE_CHECKPOINT = True
```

#### 2. Data Loading Errors
```bash
# Check file permissions
chmod -R 755 /path/to/data/

# Test NetCDF files
ncdump -h /path/to/file.nc

# Verify data structure
python -c "
import xarray as xr
ds = xr.open_dataset('/path/to/file.nc')
print(ds.variables.keys())
"
```

#### 3. Import Errors
```bash
# Reinstall requirements
pip install -r requirements.txt --force-reinstall

# Check Python path
python -c "import sys; print(sys.path)"

# Verify PyTorch installation
python -c "import torch; print(torch.__version__)"
```

#### 4. Docker Issues
```bash
# Check Docker daemon
sudo systemctl status docker

# Free up disk space
docker system prune -a

# Check GPU support in container
docker run --gpus all nvidia/cuda:11.8-base nvidia-smi
```

### Performance Optimization

#### Memory Optimization
```python
# Reduce cache size if running out of RAM
CACHE_CONFIG = {
    'max_cache_size': 50,  # Reduce from default 100
}

# Enable memory mapping for large datasets
CACHE_CONFIG = {
    'use_memory_mapping': True,
}
```

#### Speed Optimization
```python
# Increase data loading workers (if you have multiple CPU cores)
num_workers = 0  # in DataLoader

# Use mixed precision training (already enabled)
USE_AMP = True

# Pin memory for faster GPU transfer
pin_memory = True  # in DataLoader
```

##  Documentation

- **[DEVELOPMENT.md](DEVELOPMENT.md)** - Detailed developer guide
- **[data/README.md](data/README.md)** - Data structure documentation
- **API Documentation** - Coming soon
- **Model Architecture** - See source code comments

##  Contributing

We welcome contributions! Please follow these steps:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Make your changes** and add tests
4. **Run quality checks**: `pre-commit run --all-files`
5. **Commit changes**: `git commit -m 'Add amazing feature'`
6. **Push to branch**: `git push origin feature/amazing-feature`
7. **Create Pull Request**

### Development Setup
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Run tests
pytest tests/ -v --cov
```

##  License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

##  Support

- **Primary Contact**: saishashank3000@gmail.com
- **Secondary Contact**: kvsm2k@gmail.com  
- **GitHub Issues**: [Create an issue](https://github.com/BYU-Hydroinformatics/api-nwm-gcp/issues)
- **Documentation**: [GitHub Wiki](https://github.com/BYU-Hydroinformatics/api-nwm-gcp/wiki)

##  Acknowledgments

- **NRSC (National Remote Sensing Centre)** - Data and research support
- **BYU Hydroinformatics** - Infrastructure and collaboration
- **PyTorch Team** - Deep learning framework
- **NOAA/EUMETSAT** - Satellite data providers

##  Version History

- **v1.0.0** (Oct 2025) - Initial release with dual implementation
- **v0.9.0** (Sep 2025) - Beta release with SAPHIR integration  
- **v0.8.0** (Aug 2025) - Alpha release with basic UNet++ VAE

##  Roadmap

- [ ] **Real-time processing** pipeline
- [ ] **Web interface** for easy interaction
- [ ] **Multi-temporal** predictions
- [ ] **Additional satellites** integration (GPM, COSMIC)
- [ ] **Cloud deployment** (AWS, GCP, Azure)
- [ ] **Model ensemble** techniques
- [ ] **Uncertainty quantification** improvements

---

**Built with by the Atmospheric Remote Sensing Team**

> *"Predicting atmospheric humidity with unprecedented accuracy through deep learning and multi-satellite fusion"*