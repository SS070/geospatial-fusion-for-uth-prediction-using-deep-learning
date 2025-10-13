"""
Configuration for UNet++ VAE training and inference with SAPHIR integration.
Edit BASE_DIR, SAPHIR_BASE_DIR, and CKPT_DIR to match the local environment
or set the corresponding environment variables before running.
"""

import os
import torch
from pathlib import Path

# =========================
# Core data directories
# =========================
# These defaults are safe for Docker volume mounts; override via env vars.
BASE_DIR = os.getenv("MICROWAVE_DATA_DIR", "/MICROWAVE_UTH_DATA_NOAA")
SAPHIR_BASE_DIR = os.getenv("SAPHIR_DATA_DIR", "/SAPHIR_RH_DATA_PROCESSED_2")
CKPT_DIR = os.getenv("CHECKPOINT_DIR", "./checkpoints_unetpp_vae_saphir")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./outputs_with_saphir")

# Ensure directories exist (checkpoints/outputs are created if missing)
Path(CKPT_DIR).mkdir(parents=True, exist_ok=True)
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# =========================
# Satellite catalog
# =========================
# Folder names and ordering must match the training/inference dataset logic.
SATELLITE_DIRS = {
    "NOAA15": "AMSU_B_NOAA_15_1999_2002",
    "NOAA16": "AMSU_B_NOAA_16_2001_2006",
    "NOAA17": "AMSU_B_NOAA_17_2003_2008",
    "MetopA": "MHS_Metop_A_2007_2021",
    "MetopB": "MHS_Metop_B_2013_2021",
    "MetopC": "MHS_MetOp_C_2019_2021",
    "NOAA18": "MHS_NOAA_18_2006_2018",
    "NOAA19": "MHS_NOAA_19_2016_2021",
}

SATELLITE_ORDER = [
    "NOAA15", "NOAA16", "NOAA17", "MetopA", "MetopB", "MetopC", "NOAA18", "NOAA19"
]

# Quality-based weights (may be tuned)
SATELLITE_WEIGHTS = {
    "NOAA15": 0.8,
    "NOAA16": 0.85,
    "NOAA17": 0.85,
    "MetopA": 1.0,
    "MetopB": 1.0,
    "MetopC": 1.0,
    "NOAA18": 0.9,
    "NOAA19": 0.95,
}

# =========================
# SAPHIR integration
# =========================
# File pattern and latitudinal prioritization used in dataset fusion.
SAPHIR_FILE_GLOB = "uthsaphirrh*.nc"     # uthsaphirrhYYYYMMDD.nc
SAPHIR_PRIORITY_LAT_RANGE = 30.0         # ±30° band
SAPHIR_WEIGHT = float(os.getenv("SAPHIR_WEIGHT", "2.5"))  # 2.0–2.5 used in code variants
SAPHIR_MIN_COVERAGE = 0.10               # minimum valid fraction to apply SAPHIR weighting

# =========================
# File/date conventions
# =========================
# Microwave filenames embed YYYYMMDD starting at position 5 for length 8.
MICROWAVE_FILE_GLOB = "*.nc"
DATE_START_POS = 5
DATE_LENGTH = 8

# =========================
# Model configuration
# =========================
MODEL_CONFIG = {
    "in_channels": 1,
    "base": 64,
    "out_channels": 1,
    "num_sat": 8,
    "dropout": 0.10,
    "latent_dim": 512,
}

# =========================
# Training hyperparameters
# =========================
EPOCHS = int(os.getenv("EPOCHS", "25"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "1"))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "1e-4"))
GRAD_CLIP = 1.0
SAVE_EVERY = 1  # epochs

# Optimizer and scheduler
OPTIMIZER_CONFIG = {
    "lr": LEARNING_RATE,
    "betas": (0.9, 0.999),
    "eps": 1e-8,
    "weight_decay": 1e-6,
}
SCHEDULER_CONFIG = {
    "factor": 0.5,
    "patience": 3,
    "min_lr": 1e-7,
    "threshold": 1e-4,
}

# =========================
# Loss weights
# =========================
# With SAPHIR, reconstruction uses NLL/Charbonnier with SAPHIR-aware weighting.
LOSS_WEIGHTS = {
    "nll": 1.0,
    "charbonnier": 0.5,
    "kl": 0.01,
    "gradient": 0.1,
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
    print("WITH SAPHIR CONFIGURATION (UNet++ VAE)")
    print("=" * 72)
    print(f"Base microwave dir   : {BASE_DIR}")
    print(f"SAPHIR dir           : {SAPHIR_BASE_DIR}")
    print(f"Checkpoints dir      : {CKPT_DIR}")
    print(f"Outputs dir          : {OUTPUT_DIR}")
    print(f"Device / AMP         : {DEVICE} / {USE_AMP}")
    print(f"Model config         : {MODEL_CONFIG}")
    print(f"Train epochs / LR    : {EPOCHS} / {LEARNING_RATE}")
    print(f"Batch / GradClip     : {BATCH_SIZE} / {GRAD_CLIP}")
    print(f"SAPHIR weight / band : {SAPHIR_WEIGHT} / ±{SAPHIR_PRIORITY_LAT_RANGE}°")
    print("=" * 72)

if __name__ == "__main__":
    # Warn early if data roots look missing; users must edit paths or export env vars.
    missing = []
    if not os.path.isdir(BASE_DIR):
        missing.append(f"MICROWAVE_DATA_DIR not found: {BASE_DIR}")
    if not os.path.isdir(SAPHIR_BASE_DIR):
        missing.append(f"SAPHIR_DATA_DIR not found: {SAPHIR_BASE_DIR}")
    for m in missing:
        print(f"[WARNING] {m}")
    print_config()
