# UNet++ VAE with SAPHIR Integration (with_saphir)

This module provides the SAPHIR‑enabled implementation of the UNet++ VAE pipeline for gap‑filling Upper Tropospheric Humidity (UTH) using multi‑satellite microwave inputs with SAPHIR priority in the ±30° latitude band[attached_file:49][attached_file:48][attached_file:47].

## Contents

- `config.py`: Centralized configuration for data paths, model, loss, and runtime parameters; edit paths before execution or set environment variables[attached_file:47].
- `<select file to train>.py`: Select the file to train from the directory and train from entry point implementing dataset indexing, SAPHIR‑aware fusion targets, UNet++ VAE, and the SAPHIR‑aware loss with KL and gradient smoothness terms[attached_file:48][attached_file:49].
- `inference.py`: Inference pipeline with robust checkpoint loading, consistent normalization/denormalization, and optional temporal post‑processing and NetCDF export[attached_file:47].

## Data prerequisites

- Microwave inputs organized by sensor folders with filenames embedding `YYYYMMDD` starting at character position 5 for a length of 8, e.g., `xxxxxYYYYMMDDxxxxx.nc`[attached_file:49][attached_file:48].
- SAPHIR RH files named `uthsaphirrhYYYYMMDD.nc` arranged in annual subfolders under the SAPHIR base directory[attached_file:47].
- Satellite folders expected under the microwave root: `AMSU_B_NOAA_15_1999_2002`, `AMSU_B_NOAA_16_2001_2006`, `AMSU_B_NOAA_17_2003_2008`, `MHS_NOAA_18_2006_2018`, `MHS_NOAA_19_2016_2021`, `MHS_Metop_A_2007_2021`, `MHS_Metop_B_2013_2021`, and `MHS_MetOp_C_2019_2021`[attached_file:49][attached_file:48].
- The fusion target gives priority to SAPHIR within ±30° latitude using a configurable weight (default 2.5), while microwave inputs are quality‑weighted per sensor[attached_file:49][attached_file:48].

## Configuration

- Edit `BASE_DIR`, `SAPHIR_BASE_DIR`, `CKPT_DIR`, and `OUTPUT_DIR` in `config.py`, or set `MICROWAVE_DATA_DIR`, `SAPHIR_DATA_DIR`, `CHECKPOINT_DIR`, and `OUTPUT_DIR` as environment variables prior to running[attached_file:47].
- The model defaults to UNet++ with a VAE bottleneck: base channels 64, latent dimension 512, 8 satellites, and dropout 0.1[attached_file:49][attached_file:48].
- Loss components include NLL, Charbonnier, KL divergence with warm‑up, and a gradient smoothness penalty; SAPHIR regions are up‑weighted relative to microwave‑only regions[attached_file:48][attached_file:49].

## Expected directory layout

