# Development Setup Guide

## Prerequisites
- Python 3.9+
- CUDA-capable GPU (recommended)
- Docker (for containerized deployment)
- Git

## Local Development Setup

### 1. Clone Repository
```bash
git clone https://github.com/BYU-Hydroinformatics/api-nwm-gcp.git
cd api-nwm-gcp
```

### 2. Create Python Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure Data Paths
Edit the configuration files to point to your data directories:

**For SAPHIR implementation:**
```python
# with_saphir_config.py
BASE_DIR = "/path/to/your/MICROWAVE_UTH_DATA_NOAA"
SAPHIR_BASE_DIR = "/path/to/your/SAPHIR_RH_DATA_PROCESSED_2" 
```

**For non-SAPHIR implementation:**
```python  
# without_saphir_config.py
BASE_DIR = "/path/to/your/MICROWAVE_UTH_DATA_NOAA"
```

### 5. Verify Setup
```bash
# Test SAPHIR configuration
python with_saphir_config.py

# Test non-SAPHIR configuration  
python without_saphir_config.py
```

## Data Directory Structure

Ensure your data follows this structure:

```
/data/
├── MICROWAVE_UTH_DATA_NOAA/
│   ├── AMSU_B_NOAA_15_1999_2002/
│   │   ├── file_20000101_*.nc
│   │   ├── file_20000102_*.nc
│   │   └── ...
│   ├── AMSU_B_NOAA_16_2001_2006/
│   ├── AMSU_B_NOAA_17_2003_2008/
│   ├── MHS_NOAA_18_2006_2018/
│   ├── MHS_NOAA_19_2016_2021/
│   ├── MHS_Metop_A_2007_2021/
│   ├── MHS_Metop_B_2013_2021/
│   └── MHS_MetOp_C_2019_2021/
└── SAPHIR_RH_DATA_PROCESSED_2/  (optional, for SAPHIR implementation)
    ├── 2012/
    │   ├── uthsaphirrh20120101.nc
    │   ├── uthsaphirrh20120102.nc
    │   └── ...
    ├── 2013/
    ├── ...
    └── 2021/
```

## Training Models

### Option 1: With SAPHIR (Enhanced Accuracy)
```bash
cd with_saphir/
python <select file>.py
```

### Option 2: Without SAPHIR (Global Coverage)
```bash
cd without_saphir/  
python <select file>.py
```

## Running Inference

### Using Best Model
```bash
# SAPHIR implementation
python with_saphir/inference.py --checkpoint ./checkpoints_unetpp_vae_saphir/best_model.pth

# Non-SAPHIR implementation
python without_saphir/inference.py --checkpoint ./checkpoints_unetpp_vae_no_saphir/best_model.pth
```

### Single Date Prediction
```bash
python with_saphir/inference.py \
    --checkpoint ./checkpoints_unetpp_vae_saphir/best_model.pth \
    --single-date 2020-06-15 \
    --output-dir ./inference_outputs
```

## Docker Development

### Build Docker Image
```bash
# Make build script executable
chmod +x build_docker.sh

# Build image
./build_docker.sh

# Build with custom tag
./build_docker.sh --tag uth-prediction:dev
```

### Run Development Container
```bash
docker run -it --gpus all \
  -v $(pwd):/workspace \
  -v /path/to/your/data:/data \
  -v /path/to/output:/output \
  -w /workspace \
  uth-prediction:latest bash
```

## Testing

### Unit Tests
```bash
# Install test dependencies
pip install pytest pytest-cov

# Run tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

### Integration Tests
```bash
# Test data loading
python -c "
from with_saphir_train import create_combined_data_index
index = create_combined_data_index()
print(f'Found {len(index)} dates')
"

# Test model architecture
python -c "
import torch
from with_saphir_train import UNetPlusVAE
model = UNetPlusVAE()
x = torch.randn(1, 8, 1, 180, 360)
sat_ids = torch.randint(0, 8, (1, 8))
masks = torch.ones(1, 8, 180, 360)
output = model(x, sat_ids, masks)
print('Model test passed!')
"
```

## Code Style and Quality

### Pre-commit Hooks
```bash
# Install pre-commit
pip install pre-commit

# Install hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

### Code Formatting
```bash
# Format with black
black --line-length 100 *.py

# Sort imports
isort *.py

# Check with flake8
flake8 *.py --max-line-length=100
```

## Monitoring and Logging

### TensorBoard (Optional)
```bash
# Install tensorboard
pip install tensorboard

# Start tensorboard
tensorboard --logdir ./logs --port 6006

# View at http://localhost:6006
```

### Weights & Biases (Optional)
```bash
# Install wandb
pip install wandb

# Login
wandb login

# Initialize in training script
import wandb
wandb.init(project="uth-prediction")
```

## Performance Optimization

### GPU Memory Optimization
```python
# In training script, add:
torch.cuda.empty_cache()
gc.collect()

# Use gradient checkpointing for large models
# Set USE_CHECKPOINT = True in config
```

### Mixed Precision Training
```python
# Already enabled in configs
USE_AMP = True  # Automatic Mixed Precision
```

### Data Loading Optimization
```python
# Adjust cache size based on available RAM
CACHE_CONFIG = {
    'max_cache_size': 100,  # Increase if you have more RAM
    'clear_cache_interval': 50,
}
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```bash
   # Reduce batch size
   BATCH_SIZE = 1
   
   # Clear cache more frequently
   torch.cuda.empty_cache()
   ```

2. **Data Path Issues**
   ```bash
   # Check paths exist
   ls -la /path/to/MICROWAVE_UTH_DATA_NOAA/
   
   # Check permissions
   chmod -R 755 /path/to/data/
   ```

3. **Missing Dependencies**
   ```bash
   # Reinstall requirements
   pip install -r requirements.txt --force-reinstall
   ```

4. **Model Loading Errors**
   ```bash
   # Check checkpoint structure
   python -c "
   import torch
   ckpt = torch.load('checkpoint.pth', map_location='cpu')
   print(ckpt.keys())
   "
   ```

### Performance Monitoring
```bash
# Monitor GPU usage
watch nvidia-smi

# Monitor system resources
htop

# Profile Python code
pip install py-spy
py-spy top --pid <python_pid>
```

## Contributing

### Development Workflow
1. Create feature branch: `git checkout -b feature/amazing-feature`
2. Make changes and test locally
3. Run code quality checks: `pre-commit run --all-files`
4. Commit changes: `git commit -m 'Add amazing feature'`
5. Push branch: `git push origin feature/amazing-feature`
6. Create Pull Request

### Code Review Checklist
- [ ] Code follows style guidelines
- [ ] Tests pass
- [ ] Documentation updated
- [ ] Performance implications considered
- [ ] Security implications considered

## Additional Resources

- [PyTorch Documentation](https://pytorch.org/docs/)
- [NetCDF4 Python Guide](https://unidata.github.io/netcdf4-python/)
- [Xarray Documentation](http://xarray.pydata.org/)
- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)

## Support

For technical support or questions:
- Primary: saishashank3000@gmail.com
- Secondary: kvsm2k@gmail.com
- Issues: https://github.com/BYU-Hydroinformatics/api-nwm-gcp/issues