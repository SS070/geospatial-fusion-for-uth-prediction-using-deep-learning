# UNet++ VAE without SAPHIR Integration (without_saphir)

This module provides the microwave‑only implementation of the UNet++ VAE pipeline for gap‑filling Upper Tropospheric Humidity (UTH) using multi‑satellite inputs from NOAA and EUMETSAT sensors, covering a temporal range of 1999–2021[attached_file:50][attached_file:51].

## Contents

- `config.py`: Centralized configuration for data paths, model, loss, and runtime parameters; edit paths before execution or set environment variables[attached_file:50][attached_file:51].
- `<select file to train>.py`: Select the training file and traijn from entry point implementing dataset indexing, per‑satellite normalization, UNet++ VAE architecture, and composite VAE loss with optional temporal consistency and satellite weight regularization[attached_file:50][attached_file:51].

## Data prerequisites

- Microwave inputs organized by sensor folders with filenames embedding `YYYYMMDD` at character position 5 for a length of 8 (e.g., `xxxxxYYYYMMDDxxxxx.nc`)[attached_file:50][attached_file:51].
- Satellite folders expected under the microwave root: `AMSU_B_NOAA_15_1999_2002`, `AMSU_B_NOAA_16_2001_2006`, `AMSU_B_NOAA_17_2003_2008`, `MHS_NOAA_18_2006_2018`, `MHS_NOAA_19_2016_2021`, `MHS_Metop_A_2007_2021`, `MHS_Metop_B_2013_2021`, and `MHS_MetOp_C_2019_2021`[attached_file:50][attached_file:51].
- The fusion target is built from a quality‑weighted average of all available microwave satellites (no SAPHIR priority)[attached_file:50][attached_file:51].

## Configuration

- Edit `BASE_DIR`, `CKPT_DIR`, and `OUTPUT_DIR` in `config.py`, or set `MICROWAVE_DATA_DIR`, `CHECKPOINT_DIR`, and `OUTPUT_DIR` as environment variables prior to running[attached_file:50][attached_file:51].
- The model defaults to UNet++ with a VAE bottleneck: base channels 48 or 64 (depending on the variant chosen), latent dimension 512, 8 satellites, and dropout 0.1[attached_file:50][attached_file:51].
- Two loss function variants are provided:
  - **Hybrid VAE Loss**: Combines MSE, multi‑scale SSIM (MS‑SSIM), KL divergence, and total variation regularization[attached_file:50].
  - **Advanced Composite VAE Loss**: Combines NLL, Charbonnier, SSIM (optional), KL divergence, gradient smoothness, temporal consistency (optional), and satellite weight regularization; all components are constrained to non‑negative values for stable training[attached_file:51].

## Expected directory layout

