"""
Configuration for UNet++ VAE training and inference without SAPHIR integration.
Microwave‑only implementation for global coverage (1999–2021).

Edit BASE_DIR and CKPT_DIR to match the local environment or set the corresponding
environment variables before running.
"""

import os
import torch
from pathlib import Path

# =========================
# Core data directory
# =========================
# Safe default for Docker volume mounts; override via environment variable.
BASE_DIR = os.getenv("MICROWAVE_DATA_DIR", "/MICROWAVE_UTH_DATA_NOAA")
CKPT_DIR = os.getenv("CHECKPOINT_DIR", "./checkpoints_unetpp_vae")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./outputs_without_saphir")

# Ensure directories exist
Path(CKPT_DIR).mkdir(parents=True, exist_ok=True)
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# =========================
# Satellite catalog
# =========================
# Folder names must match the dataset file structure.
SATELLITE_DIRS = {
    "NOAA_15": "AMSU_B_NOAA_15_1999_2002",
    "NOAA_16": "AMSU_B_NOAA_16_2001_2006",
    "NOAA_17": "AMSU_B_NOAA_17_2003_2008",
    "Metop_A": "MHS_Metop_A_2007_2021",
    "Metop_B": "MHS_Metop_B_2013_2021",
    "Metop_C": "MHS_MetOp_C_2019_2021",
    "NOAA_18": "MHS_NOAA_18_2006_2018",
    "NOAA_19": "MHS_NOAA_19_2016_2021",
}

SATELLITE_ORDER = [
    "NOAA_15", "NOAA_16", "NOAA_17", "Metop_A",
    "Metop_B", "Metop_C", "NOAA_18", "NOAA_19"
]

# Satellite to index mapping for model input
SATELLITE_TO_ID = {sat: idx for idx, sat in enumerate(SATELLITE_ORDER)}

# =========================
# File and date conventions
# =========================
# Microwave filenames embed YYYYMMDD starting at character position 5 for length 8.
MICROWAVE_FILE_GLOB = "*.nc"
DATE_START_POS = 5
DATE_LENGTH = 8

# =========================
# Model configuration
# =========================
MODEL_CONFIG = {
    "in_channels": 1,
    "base": 64,           # 48 or 64 depending on variant
    "out_channels": 1,
    "num_sat": 8,         # 8 microwave satellites
    "dropout": 0.10,
    "latent_dim": 512,    # VAE bottleneck latent dimension
}

# =========================
# Training hyperparameters
# =========================
EPOCHS = int(os.getenv("EPOCHS", "25"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "1"))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "1e-4"))
GRAD_CLIP = 1.0
SAVE_EVERY = 5  # epochs

# Optimizer and scheduler
OPTIMIZER_CONFIG = {
    "lr": LEARNING_RATE,
    "betas": (0.9, 0.999),
    "eps": 1e-8,
    "weight_decay": 1e-5,
}
SCHEDULER_CONFIG = {
    "factor": 0.5,
    "patience": 5,
    "min_lr": 1e-7,
    "threshold": 1e-4,
}

# =========================
# Loss weights (variant A: hybrid VAE loss)
# =========================
LOSS_WEIGHTS_HYBRID = {
    "mse_alpha": 0.6,
    "ssim_beta": 0.4,
    "kl_gamma_max": 0.01,
    "tv_weight": 5e-3,
    "kl_warmup_epochs": 10,
}

# =========================
# Loss weights (variant B: advanced composite loss)
# =========================
LOSS_WEIGHTS_ADVANCED = {
    "lambda_nll": 1.0,
    "lambda_char": 0.5,
    "lambda_ssim": 0.0,       # set > 0 to enable SSIM component
    "lambda_grad": 0.05,
    "beta_kl": 0.01,
    "lambda_temp": 0.25,
    "lambda_sat_reg": 1e-3,
    "charbonnier_eps": 1e-3,
    "logvar_clamp": (-10.0, 10.0),
    "warmup_epochs": 5,
    "kl_warmup_epochs": 10,
    "temporal_decay": 0.995,
}

# =========================
# Data processing and caching
# =========================
NORMALIZATION = {
    "method": "iqr",
    "clip_outliers": True,
    "fill_value": 0.0,
}
CACHE_CONFIG = {
    "max_cache_size": 150,
    "clear_cache_interval": 50,
    "use_memory_mapping": True,
}

# =========================
# Validation split
# =========================
VALIDATION = {
    "split_ratio": 0.10,
    "shuffle_data": True,
    "random_seed": 42,
}

# =========================
# Device and AMP
# =========================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = torch.cuda.is_available()

def print_config() -> None:
    print("=" * 72)
    print("WITHOUT SAPHIR CONFIGURATION (UNet++ VAE)")
    print("=" * 72)
    print(f"Base microwave dir   : {BASE_DIR}")
    print(f"Checkpoints dir      : {CKPT_DIR}")
    print(f"Outputs dir          : {OUTPUT_DIR}")
    print(f"Device / AMP         : {DEVICE} / {USE_AMP}")
    print(f"Model config         : {MODEL_CONFIG}")
    print(f"Train epochs / LR    : {EPOCHS} / {LEARNING_RATE}")
    print(f"Batch / GradClip     : {BATCH_SIZE} / {GRAD_CLIP}")
    print("=" * 72)

if __name__ == "__main__":
    # Warn if data root looks missing; users must edit paths or export env vars.
    if not os.path.isdir(BASE_DIR):
        print(f"[WARNING] MICROWAVE_DATA_DIR not found: {BASE_DIR}")
    print_config()
