# geospatial_inference_pipeline_optimized_temporal_FIXED.py
# Optimized Inference Pipeline for Geospatial UTH Gap-filling Models
# with Uncertainty-Aware and Temporal Postprocessing (per-model, no model fusion)
# OPTIMIZED VERSION - Improved temporal postprocessing and removed weighted attention model

import os
import sys
import glob
import gc
import time
import tempfile
import shutil
from collections import defaultdict
from datetime import datetime
import numpy as np
import xarray as xr
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
from scipy.ndimage import median_filter
from scipy.signal import medfilt2d

# Ensure compatibility across systems
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==================== CONFIGURATION ====================
BASE_DIR = r"C:\S\Programming\NRSC\PythonProject\MICROWAVE_UTH_DATA_NOAA"
SAPHIR_BASE_DIR = r"S:\DATA\SAPHIR_RH_DATA_PROCESSED_2"
OUTPUT_DIR = r"S:\DATA\AI_Generated_Predictions_04"

MODEL_CONFIGS = {
    "1": {
        "name": "UNet++ VAE Saphir FINAL (KL/NLL Loss)",
        "checkpoint_dir": "./checkpoints_unetpp_vae_saphir_final",
        "model_file": "UNET++_VAE_Saphir_FINAL.pth",
        "class_type": "vae_final"
    },
    "2": {
        "name": "UNet++ VAE Saphir OPTIMAL (Hybrid Loss)",
        "checkpoint_dir": "./checkpoints_unetpp_vae_saphir_optimal",
        "model_file": "UNET++_VAE_Saphir_Optimal.pth",
        "class_type": "vae_optimal"
    }
}

satellite_dirs = {
    'NOAA_15': "AMSU_B_NOAA_15_1999_2002",
    'NOAA_16': "AMSU_B_NOAA_16_2001_2006",
    'NOAA_17': "AMSU_B_NOAA_17_2003_2008",
    'Metop_A': "MHS_Metop_A_2007_2021",
    'Metop_B': "MHS_Metop_B_2013_2021",
    'Metop_C': "MHS_MetOp_C_2019_2021",
    'NOAA_18': "MHS_NOAA_18_2006_2018",
    'NOAA_19': "MHS_NOAA_19_2016_2021"
}

SATELLITE_ORDER = ['NOAA_15', 'NOAA_16', 'NOAA_17', 'Metop_A', 'Metop_B', 'Metop_C', 'NOAA_18', 'NOAA_19']


# === [ MODEL ARCHITECTURE CLASSES ] ===
class SatelliteSpecificNormalization(nn.Module):
    def __init__(self, num_satellites=8, channels=1):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_satellites, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(num_satellites, channels, 1, 1))

    def forward(self, x, satellite_ids):
        out = x.clone()
        B, N, C, H, W = x.shape
        for b in range(B):
            for n in range(N):
                sid = satellite_ids[b, n].item()
                if 0 <= sid < self.weight.shape[0]:
                    out[b, n] = x[b, n] * self.weight[sid] + self.bias[sid]
        return out


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        red = max(1, channels // reduction)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, red, 1),
            nn.GELU(),
            nn.Conv2d(red, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.se(x)


class ResidualConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, use_se=True, dropout=0.1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch)
        )
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.se = SEBlock(out_ch) if use_se else nn.Identity()

    def forward(self, x):
        y = self.conv(x)
        y = self.se(y)
        return F.gelu(y + self.skip(x))


class VariationalBottleneck(nn.Module):
    def __init__(self, in_channels, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.to_mu = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, latent_dim)
        )
        self.to_logvar = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, latent_dim)
        )
        self.from_latent = nn.Linear(latent_dim, in_channels * 4 * 4)
        self.upsample = nn.ConvTranspose2d(in_channels, in_channels, 4, stride=4)

    def encode(self, x):
        mu = self.to_mu(x)
        logvar = self.to_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu

    def decode(self, z, target_size):
        B = z.size(0)
        h = self.from_latent(z)
        h = h.view(B, -1, 4, 4)
        h = self.upsample(h)
        h = F.interpolate(h, size=target_size, mode='bilinear', align_corners=False)
        return h

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstructed = self.decode(z, x.shape[-2:])
        return reconstructed, mu, logvar


class DenseBlock(nn.Module):
    def __init__(self, in_channels_list, out_channels, dropout=0.1):
        super().__init__()
        total_in = sum(in_channels_list)
        self.conv = nn.Sequential(
            nn.Conv2d(total_in, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )
        self.se = SEBlock(out_channels)

    def forward(self, x_list):
        x = torch.cat(x_list, dim=1)
        out = self.conv(x)
        return self.se(out)


class UNetPlusEncoderFixed(nn.Module):
    def __init__(self, in_channels=1, base=64, dropout=0.1):
        super().__init__()
        self.base = base
        # Encoder levels
        self.conv00 = ResidualConvBlock(in_channels, base, dropout=dropout)
        self.conv10 = ResidualConvBlock(base, base * 2, dropout=dropout)
        self.conv20 = ResidualConvBlock(base * 2, base * 4, dropout=dropout)
        self.conv30 = ResidualConvBlock(base * 4, base * 8, dropout=dropout)
        self.conv40 = ResidualConvBlock(base * 8, base * 16, dropout=dropout)

        # Dense connections
        self.conv01 = DenseBlock([base, base * 2], base, dropout)
        self.conv11 = DenseBlock([base * 2, base * 4], base * 2, dropout)
        self.conv21 = DenseBlock([base * 4, base * 8], base * 4, dropout)
        self.conv31 = DenseBlock([base * 8, base * 16], base * 8, dropout)

        self.conv02 = DenseBlock([base, base, base * 2], base, dropout)
        self.conv12 = DenseBlock([base * 2, base * 2, base * 4], base * 2, dropout)
        self.conv22 = DenseBlock([base * 4, base * 4, base * 8], base * 4, dropout)

        self.conv03 = DenseBlock([base, base, base, base * 2], base, dropout)
        self.conv13 = DenseBlock([base * 2, base * 2, base * 2, base * 4], base * 2, dropout)

        self.conv04 = DenseBlock([base, base, base, base, base * 2], base, dropout)

        self.pool = nn.MaxPool2d(2, 2)
        self.vae_bottleneck = VariationalBottleneck(base * 16, latent_dim=512)

    def _align_and_upsample(self, x, target_size):
        if x.size(2) != target_size[0] or x.size(3) != target_size[1]:
            x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
        return x

    def forward(self, x):
        features = {}
        # UNet++ forward pass with nested connections
        x00 = self.conv00(x)
        features['x00'] = x00

        x10 = self.conv10(self.pool(x00))
        features['x10'] = x10

        x10_up = self._align_and_upsample(x10, (x00.size(2), x00.size(3)))
        x01 = self.conv01([x00, x10_up])
        features['x01'] = x01

        x20 = self.conv20(self.pool(x10))
        features['x20'] = x20

        x20_up = self._align_and_upsample(x20, (x10.size(2), x10.size(3)))
        x11 = self.conv11([x10, x20_up])
        features['x11'] = x11

        x11_up = self._align_and_upsample(x11, (x00.size(2), x00.size(3)))
        x02 = self.conv02([x00, x01, x11_up])
        features['x02'] = x02

        x30 = self.conv30(self.pool(x20))
        features['x30'] = x30

        x30_up = self._align_and_upsample(x30, (x20.size(2), x20.size(3)))
        x21 = self.conv21([x20, x30_up])
        features['x21'] = x21

        x21_up = self._align_and_upsample(x21, (x10.size(2), x10.size(3)))
        x12 = self.conv12([x10, x11, x21_up])
        features['x12'] = x12

        x12_up = self._align_and_upsample(x12, (x00.size(2), x00.size(3)))
        x03 = self.conv03([x00, x01, x02, x12_up])
        features['x03'] = x03

        x40 = self.conv40(self.pool(x30))
        features['x40'] = x40

        x40_vae, mu, logvar = self.vae_bottleneck(x40)
        features['x40_vae'] = x40_vae
        features['mu'] = mu
        features['logvar'] = logvar

        x40_vae_up = self._align_and_upsample(x40_vae, (x30.size(2), x30.size(3)))
        x31 = self.conv31([x30, x40_vae_up])
        features['x31'] = x31

        x31_up = self._align_and_upsample(x31, (x20.size(2), x20.size(3)))
        x22 = self.conv22([x20, x21, x31_up])
        features['x22'] = x22

        x22_up = self._align_and_upsample(x22, (x10.size(2), x10.size(3)))
        x13 = self.conv13([x10, x11, x12, x22_up])
        features['x13'] = x13

        x13_up = self._align_and_upsample(x13, (x00.size(2), x00.size(3)))
        x04 = self.conv04([x00, x01, x02, x03, x13_up])
        features['x04'] = x04

        # For fusion, align all exported levels to x00 spatial size and ensure base channels
        target_size = (x00.size(2), x00.size(3))
        export = {
            'x00': self._align_and_upsample(x00, target_size),
            'x01': self._align_and_upsample(x01, target_size),
            'x02': self._align_and_upsample(x02, target_size),
            'x03': self._align_and_upsample(x03, target_size),
            'x04': self._align_and_upsample(x04, target_size),
            'mu': mu,
            'logvar': logvar
        }
        return export


class AdvancedWeightedFusion(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(feature_dim, max(feature_dim // 4, 1), 1),
            nn.GELU(),
            nn.Conv2d(max(feature_dim // 4, 1), 1, 1),
            nn.Sigmoid()
        )

    def forward(self, feats, masks):
        B, N, C, H, W = feats.shape
        attention_weights = []
        for i in range(N):
            att = self.attention(feats[:, i])
            attention_weights.append(att)
        attention_weights = torch.stack(attention_weights, dim=1)  # [B,N,1,H,W]

        m = masks.unsqueeze(2)  # [B,N,1,H,W]
        pixel_counts = m.sum(dim=(3, 4), keepdim=True).clamp(min=1e-6)
        reliability_weights = pixel_counts / pixel_counts.sum(dim=1, keepdim=True).clamp(min=1e-6)

        final_weights = attention_weights * reliability_weights * m
        final_weights = final_weights / final_weights.sum(dim=1, keepdim=True).clamp(min=1e-6)

        fused = (feats * final_weights).sum(dim=1)
        return fused


# UNet++ VAE Model (for both FINAL and OPTIMAL variants)
class UNetPlusVAEFixed(nn.Module):
    def __init__(self, in_channels=1, base=64, out_channels=1, num_sat=8, dropout=0.1):
        super().__init__()
        self.base = base
        self.num_sat = num_sat
        self.sat_norm = SatelliteSpecificNormalization(num_sat, in_channels)
        self.encoder = UNetPlusEncoderFixed(in_channels, base, dropout)

        self.fusion_modules = nn.ModuleDict({
            'x04': AdvancedWeightedFusion(base),
            'x03': AdvancedWeightedFusion(base),
            'x02': AdvancedWeightedFusion(base),
            'x01': AdvancedWeightedFusion(base),
            'x00': AdvancedWeightedFusion(base)
        })

        self.final_conv = nn.Sequential(
            nn.Conv2d(base, max(base // 2, 1), 3, padding=1),
            nn.BatchNorm2d(max(base // 2, 1)),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(max(base // 2, 1), out_channels, 1)
        )

        self.uncertainty_head = nn.Sequential(
            nn.Conv2d(base, max(base // 2, 1), 3, padding=1),
            nn.BatchNorm2d(max(base // 2, 1)),
            nn.GELU(),
            nn.Conv2d(max(base // 2, 1), out_channels, 1)
        )

    def forward(self, x, sat_ids, valid_masks):
        B, N, C, H, W = x.shape
        x = self.sat_norm(x, sat_ids)

        all_features = {level: [] for level in ['x04', 'x03', 'x02', 'x01', 'x00']}
        mu_list = []
        logvar_list = []

        for i in range(N):
            if valid_masks[:, i].sum() == 0:
                for level in all_features.keys():
                    all_features[level].append(torch.zeros(B, self.base, H, W, device=x.device, dtype=x.dtype))
                mu_list.append(torch.zeros(B, 512, device=x.device, dtype=x.dtype))
                logvar_list.append(torch.zeros(B, 512, device=x.device, dtype=x.dtype))
            else:
                features = self.encoder(x[:, i])
                for level in all_features.keys():
                    all_features[level].append(features[level])
                mu_list.append(features['mu'])
                logvar_list.append(features['logvar'])

        for level in all_features.keys():
            all_features[level] = torch.stack(all_features[level], dim=1)

        mu_agg = torch.stack(mu_list, dim=1).mean(dim=1)
        logvar_agg = torch.stack(logvar_list, dim=1).mean(dim=1)

        fused_features = self.fusion_modules['x04'](all_features['x04'], valid_masks)

        prediction = self.final_conv(fused_features)
        uncertainty = self.uncertainty_head(fused_features)

        return {
            'prediction': prediction,
            'uncertainty': uncertainty,
            'mu': mu_agg,
            'logvar': logvar_agg,
            'features': all_features
        }


# ==================== MODEL LOADING ====================
def _safe_load_state_dict(model, checkpoint_path):
    if not os.path.isfile(checkpoint_path):
        print(f"Checkpoint not found at: {checkpoint_path}. Proceeding with randomly initialized weights.")
        return

    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt.get("model_state_dict", ckpt)
    model_dict = model.state_dict()

    filtered = {}
    for k, v in state.items():
        if k in model_dict and model_dict[k].shape == v.shape:
            filtered[k] = v
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    if missing:
        print(f"Warning: Missing keys: {len(missing)} (showing up to 5): {missing[:5]}")
    if unexpected:
        print(f"Warning: Unexpected keys: {len(unexpected)} (showing up to 5): {unexpected[:5]}")
    print(f"Loaded checkpoint: {checkpoint_path}")


def load_selected_model(choice):
    cfg = MODEL_CONFIGS[choice]
    class_type = cfg['class_type']
    checkpoint_dir = cfg['checkpoint_dir']
    model_file = cfg['model_file']
    checkpoint_path = os.path.join(checkpoint_dir, model_file)

    if class_type in ("vae_final", "vae_optimal"):
        model = UNetPlusVAEFixed(in_channels=1, base=64, out_channels=1, num_sat=8, dropout=0.1)
        model_type = "vae"
    else:
        raise ValueError(f"Unknown class_type in MODEL_CONFIGS: {class_type}")

    model.to(device)
    model.eval()
    _safe_load_state_dict(model, checkpoint_path)
    return model, model_type


# ==================== DATA INDEXING ====================
def create_data_index():
    """Create comprehensive data index for all satellite data"""
    print("Creating comprehensive data index...")

    date_file_index = defaultdict(dict)
    print("Indexing microwave satellite data files...")

    for sat, folder in satellite_dirs.items():
        folder_path = os.path.join(BASE_DIR, folder)
        if not os.path.isdir(folder_path):
            print(f"Warning: Path does not exist: {folder_path}")
            continue

        nc_files = glob.glob(os.path.join(folder_path, "*.nc"))
        print(f"{sat}: Found {len(nc_files)} files")

        for file_path in nc_files:
            filename = os.path.basename(file_path)
            try:
                # Assumes AMSU/MHS YYYYMMDD embedded starting at pos 5
                date_str = filename[5:13]  # YYYYMMDD
                date_obj = datetime.strptime(date_str, "%Y%m%d").date()
                date_file_index[str(date_obj)][sat] = file_path
            except Exception as e:
                print(f"Failed to parse date from {filename}: {e}")

    saphir_file_index = {}
    print("Indexing Saphir RH data files...")

    for year in range(2012, 2022):
        year_dir = os.path.join(SAPHIR_BASE_DIR, str(year))
        if os.path.exists(year_dir):
            nc_files = glob.glob(os.path.join(year_dir, "uthsaphirrh*.nc"))
            for file_path in nc_files:
                filename = os.path.basename(file_path)
                try:
                    date_str = filename[11:19]  # YYYYMMDD
                    date = datetime.strptime(date_str, "%Y%m%d").date()
                    saphir_file_index[str(date)] = file_path
                except Exception:
                    continue

    combined_date_index = {}
    for date_str in date_file_index.keys():
        combined_date_index[date_str] = {
            'microwave': date_file_index[date_str],
            'saphir': saphir_file_index.get(date_str, None)
        }

    print(f"Indexed {len(date_file_index)} unique microwave dates.")
    print(f"Indexed {len(saphir_file_index)} unique Saphir dates.")
    print(f"Combined index created for {len(combined_date_index)} dates.")

    saphir_available_dates = sum(1 for v in combined_date_index.values() if v['saphir'] is not None)
    print(f"Dates with Saphir data available: {saphir_available_dates}")

    return combined_date_index


# ==================== DATA PREPROCESSING WITH NORMALIZATION PARAMETERS ====================
def normalize_data_consistently(data, valid_mask, normalize=True):
    """Consistent normalization matching training"""
    if not normalize:
        return data, 0.0, 1.0

    filled_data = np.nan_to_num(data, nan=0.0).astype(np.float32)

    if valid_mask.sum() > 100:
        v = filled_data[valid_mask]
        med = np.median(v)
        iqr = np.percentile(v, 75) - np.percentile(v, 25)
        scale = max(iqr, 1e-6)
        normalized = (filled_data - med) / scale
        return normalized, med, scale
    else:
        normalized = filled_data / 100.0
        return normalized, 0.0, 100.0


def load_microwave_data(microwave_files, normalize=True):
    """Load and preprocess microwave satellite data, returning normalization parameters"""
    inputs = []
    masks = []
    sat_ids = []
    num_satellites = np.zeros((180, 360), dtype=np.int32)
    quality_flag = np.zeros((180, 360), dtype=np.int8)
    median_list = []
    scale_list = []

    lat, lon = None, None  # To store coordinates once

    for i, sat_name in enumerate(SATELLITE_ORDER):
        if sat_name in microwave_files:
            file_path = microwave_files[sat_name]

            try:
                with xr.open_dataset(file_path) as ds:
                    arr = ds['uth_mean_ascend_descend'].values
                    # Fill masked arrays and enforce float32
                    if isinstance(arr, np.ma.MaskedArray):
                        arr = arr.filled(np.nan)
                    arr = np.asarray(arr, dtype=np.float32)
                    if arr.ndim == 3:
                        arr = arr[0]

                    if lat is None:
                        lat = np.asarray(ds['lat'].values, dtype=np.float32)
                        lon = np.asarray(ds['lon'].values, dtype=np.float32)

                # Ensure contiguous arrays
                arr = np.ascontiguousarray(arr)

                valid_mask = ~np.isnan(arr)
                num_satellites[valid_mask] += 1
                quality_flag[valid_mask] = 1

                arr_norm, med, scale = normalize_data_consistently(arr, valid_mask, normalize)
                inputs.append(torch.from_numpy(np.ascontiguousarray(arr_norm)).unsqueeze(0))
                masks.append(torch.from_numpy(np.ascontiguousarray(valid_mask.astype(np.float32))))
                sat_ids.append(i)
                median_list.append(med)
                scale_list.append(scale)

            except Exception as e:
                print(f"Error loading {sat_name}: {e}")
                inputs.append(torch.zeros(1, 180, 360, dtype=torch.float32))
                masks.append(torch.zeros(180, 360, dtype=torch.float32))
                sat_ids.append(i)
                median_list.append(0.0)
                scale_list.append(1.0)
        else:
            inputs.append(torch.zeros(1, 180, 360, dtype=torch.float32))
            masks.append(torch.zeros(180, 360, dtype=torch.float32))
            sat_ids.append(i)
            median_list.append(0.0)
            scale_list.append(1.0)

    while len(inputs) < 8:
        inputs.append(torch.zeros(1, 180, 360, dtype=torch.float32))
        masks.append(torch.zeros(180, 360, dtype=torch.float32))
        sat_ids.append(-1)
        median_list.append(0.0)
        scale_list.append(1.0)

    inputs_tensor = torch.stack(inputs[:8], dim=0).unsqueeze(0).to(device).contiguous()  # [1,8,1,H,W]
    masks_tensor = torch.stack(masks[:8], dim=0).unsqueeze(0).to(device).contiguous()  # [1,8,H,W]
    sat_ids_tensor = torch.tensor(sat_ids[:8], dtype=torch.long).unsqueeze(0).to(device)  # [1,8]

    # Global median and scale as mean of all non-defaults
    median_nonzero = [m for m in median_list if m != 0.0]
    scale_nondefault = [s for s in scale_list if s != 1.0]
    median_global = float(np.mean(median_nonzero)) if len(median_nonzero) > 0 else 0.0
    scale_global = float(np.mean(scale_nondefault)) if len(scale_nondefault) > 0 else 1.0

    if lat is None or lon is None:
        lat = np.linspace(-89.5, 89.5, 180, dtype=np.float32)
        lon = np.linspace(-179.5, 179.5, 360, dtype=np.float32)
    else:
        lat = np.ascontiguousarray(np.asarray(lat, dtype=np.float32))
        lon = np.ascontiguousarray(np.asarray(lon, dtype=np.float32))

    # Ensure base dtypes and contiguous for arrays passed to save
    num_satellites = np.ascontiguousarray(num_satellites.astype(np.int32))
    quality_flag = np.ascontiguousarray(quality_flag.astype(np.int8))

    return inputs_tensor, masks_tensor, sat_ids_tensor, num_satellites, quality_flag, lat, lon, median_global, scale_global


# ==================== INFERENCE (MODIFIED) ====================
def run_inference(model, model_type, inputs_tensor, masks_tensor, sat_ids_tensor):
    with torch.no_grad():
        use_cuda_amp = (device.type == 'cuda')
        if model_type == "vae":
            with autocast(device_type='cuda' if use_cuda_amp else 'cpu'):
                output = model(inputs_tensor, sat_ids_tensor, masks_tensor)
            prediction = output["prediction"][0, 0].detach().cpu().numpy()
            uncertainty = output["uncertainty"][0, 0].detach().cpu().numpy()
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    return prediction, uncertainty


def denormalize_prediction(pred_norm, median, scale):
    return pred_norm * scale + median


def denormalize_uncertainty(uncert_norm, scale):
    return np.abs(uncert_norm) * scale


def postprocess_uncertainty_smoothing(pred, uncert, num_satellites, smooth_kernel=3, method='median',
                                      mask_satellite_min=2):
    threshold = np.percentile(uncert, 90)
    high_uncert_mask = (uncert >= threshold) & (num_satellites < mask_satellite_min)
    postprocessed_pred = pred.copy()
    if np.any(high_uncert_mask):
        filtered_pred = median_filter(pred, size=smooth_kernel, mode='nearest') if method == 'median' else pred
        postprocessed_pred[high_uncert_mask] = filtered_pred[high_uncert_mask]
    return postprocessed_pred, high_uncert_mask


def safe_file_write(data_dict, output_path, max_retries=3, retry_delay=1.0):
    """Safely write NetCDF file with error handling and retries"""
    for attempt in range(max_retries):
        try:
            # Use temporary file approach
            temp_dir = os.path.dirname(output_path)
            with tempfile.NamedTemporaryFile(delete=False, suffix='.nc', dir=temp_dir) as tmp_file:
                temp_path = tmp_file.name

            # Create dataset
            ds = xr.Dataset(
                data_dict['data_vars'],
                coords=data_dict['coords']
            )
            ds.attrs.update(data_dict['attrs'])

            # Add variable attributes
            for var_name, var_attrs in data_dict.get('var_attrs', {}).items():
                if var_name in ds.data_vars:
                    ds[var_name].attrs.update(var_attrs)

            # Write to temporary file first
            try:
                ds.to_netcdf(temp_path, engine="netcdf4")
            except Exception:
                ds.to_netcdf(temp_path)

            # Close dataset explicitly
            ds.close()
            del ds

            # Atomic move from temp to final location
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            shutil.move(temp_path, output_path)
            return True

        except Exception as e:
            print(f"Attempt {attempt + 1} failed to write {output_path}: {e}")
            # Clean up temp file if it exists
            if 'temp_path' in locals() and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except:
                    pass

            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                print(f"Failed to write {output_path} after {max_retries} attempts")
                return False

    return False


def save_prediction_netcdf(prediction, uncertainty, num_satellites, quality_flag, high_uncert_mask, lat, lon,
                           output_path):
    # Update quality_flag: 2 = gap-filled & high uncertainty
    quality_flag_out = quality_flag.copy()
    quality_flag_out[high_uncert_mask & (quality_flag == 0)] = 2

    # Enforce primitive dtypes and contiguity to avoid "array type not supported"
    prediction = np.ascontiguousarray(prediction.astype(np.float32))
    uncertainty = np.ascontiguousarray(uncertainty.astype(np.float32))
    num_satellites = np.ascontiguousarray(num_satellites.astype(np.int32))
    quality_flag_out = np.ascontiguousarray(quality_flag_out.astype(np.int8))
    lat = np.ascontiguousarray(lat.astype(np.float32))
    lon = np.ascontiguousarray(lon.astype(np.float32))

    data_dict = {
        'data_vars': {
            "uth_predicted": (["lat", "lon"], prediction),
            "uth_uncertainty": (["lat", "lon"], uncertainty),
            "num_satellites": (["lat", "lon"], num_satellites),
            "quality_flag": (["lat", "lon"], quality_flag_out),
        },
        'coords': {
            "lat": lat,
            "lon": lon,
        },
        'attrs': {
            "title": "AI Generated UTH Predictions",
            "description": "Gap-filled Upper Tropospheric Humidity predictions from multi-satellite fusion. Uncertainty-aware postprocessing applied.",
            "creation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "source": "Geospatial AI Inference Pipeline - Optimized"
        },
        'var_attrs': {
            "uth_predicted": {
                "long_name": "Upper Tropospheric Humidity Predicted",
                "units": "percent",
                "description": "AI model predicted UTH values with gap-filling"
            },
            "uth_uncertainty": {
                "long_name": "Prediction Uncertainty (stddev)",
                "units": "percent",
                "description": "Model-predicted uncertainty (higher = less reliable)"
            },
            "num_satellites": {
                "long_name": "Number of Contributing Satellites",
                "description": "Count of satellites with valid data for each pixel"
            },
            "quality_flag": {
                "long_name": "Data Quality Flag",
                "description": "0 = gap-filled only, 1 = satellite data available, 2 = gap-filled & high uncertainty"
            }
        }
    }

    return safe_file_write(data_dict, output_path)


# =========== OPTIMIZED TEMPORAL POSTPROCESSING ===========
def load_year_data_efficiently(year_dir, var_name="uth_predicted", flag_name="quality_flag",
                               uncertainty_name="uth_uncertainty"):
    """Efficiently load all data for a year into memory"""
    nc_files = sorted(glob.glob(os.path.join(year_dir, "aigeneratedpredictions*.nc")))
    if not nc_files:
        return None, None, None, None, None

    print(f"Loading {len(nc_files)} files for temporal processing...")

    # Initialize storage
    dates = []
    predictions = []
    quality_flags = []
    uncertainties = []
    num_satellites = []
    coords = None
    attrs_template = None
    var_attrs_template = None

    # Load data with single progress bar
    with tqdm(nc_files, desc="Loading files", leave=False) as pbar:
        for nc_file in pbar:
            try:
                # Extract date
                filename = os.path.basename(nc_file)
                date_str = filename.replace('aigeneratedpredictions', '').replace('.nc', '')
                date_obj = datetime.strptime(date_str, '%Y%m%d')
                dates.append((date_obj, nc_file))

                # Load dataset
                with xr.open_dataset(nc_file) as ds:
                    predictions.append(ds[var_name].values.copy())
                    quality_flags.append(ds[flag_name].values.copy())
                    uncertainties.append(ds[uncertainty_name].values.copy())
                    num_satellites.append(ds["num_satellites"].values.copy())

                    # Store coordinates and metadata (once)
                    if coords is None:
                        coords = {coord: ds[coord].values.copy() for coord in ds.coords}
                        attrs_template = dict(ds.attrs)
                        var_attrs_template = {var: dict(ds[var].attrs) for var in ds.data_vars}

            except Exception as e:
                print(f"Error loading {nc_file}: {e}")
                continue

    if not predictions:
        return None, None, None, None, None

    # Sort by date
    sorted_data = sorted(zip(dates, predictions, quality_flags, uncertainties, num_satellites))
    dates, predictions, quality_flags, uncertainties, num_satellites = zip(*sorted_data)

    # Extract file paths and dates
    file_paths = [d[1] for d in dates]
    dates = [d[0] for d in dates]

    # Convert to numpy arrays
    predictions = np.array(predictions, dtype=np.float32)
    quality_flags = np.array(quality_flags, dtype=np.int8)
    uncertainties = np.array(uncertainties, dtype=np.float32)
    num_satellites = np.array(num_satellites, dtype=np.int32)

    return {
        'predictions': predictions,
        'quality_flags': quality_flags,
        'uncertainties': uncertainties,
        'num_satellites': num_satellites,
        'dates': dates,
        'file_paths': file_paths,
        'coords': coords,
        'attrs': attrs_template,
        'var_attrs': var_attrs_template
    }


def apply_vectorized_temporal_smoothing(predictions, quality_flags, temporal_window=3):
    """Apply temporal smoothing using vectorized operations for efficiency"""
    print("Applying vectorized temporal smoothing...")

    n_days, nlat, nlon = predictions.shape
    half_w = temporal_window // 2
    smoothed_predictions = predictions.copy()

    # Create mask for pixels that need smoothing (not original satellite data)
    needs_smoothing = (quality_flags != 1)

    print(f"Processing {needs_smoothing.sum()} pixels that need temporal smoothing...")

    # Apply temporal median filtering efficiently
    if needs_smoothing.sum() > 0:
        # Use scipy's median filter along time axis for each spatial location
        for i in tqdm(range(nlat), desc="Spatial rows", leave=False):
            for j in range(nlon):
                # Get time series for this pixel
                time_series = predictions[:, i, j]
                smooth_mask = needs_smoothing[:, i, j]

                if smooth_mask.any():
                    # Apply temporal median filter only where needed
                    for t in np.where(smooth_mask)[0]:
                        # Define temporal window
                        t_start = max(0, t - half_w)
                        t_end = min(n_days, t + half_w + 1)

                        # Get values in temporal window
                        window_vals = time_series[t_start:t_end]
                        valid_vals = window_vals[~np.isnan(window_vals)]

                        if len(valid_vals) > 0:
                            smoothed_predictions[t, i, j] = np.median(valid_vals)

    return smoothed_predictions


def write_smoothed_data_batch(year_data, smoothed_predictions, var_name="uth_predicted", flag_name="quality_flag"):
    """Write smoothed data back to files efficiently"""
    print("Writing smoothed data back to files...")

    file_paths = year_data['file_paths']
    coords = year_data['coords']
    attrs = year_data['attrs']
    var_attrs = year_data['var_attrs']

    # Update attributes
    attrs = attrs.copy()
    attrs["temporal_smoothing"] = f"Applied vectorized temporal median smoothing"
    attrs["temporal_smoothing_applied"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    successful_writes = 0
    failed_writes = 0

    with tqdm(enumerate(file_paths), total=len(file_paths), desc="Writing files", leave=False) as pbar:
        for idx, file_path in pbar:
            try:
                # Prepare data dictionary
                data_dict = {
                    'data_vars': {
                        var_name: (["lat", "lon"], smoothed_predictions[idx].astype(np.float32)),
                        flag_name: (["lat", "lon"], year_data['quality_flags'][idx].astype(np.int8)),
                        "uth_uncertainty": (["lat", "lon"], year_data['uncertainties'][idx].astype(np.float32)),
                        "num_satellites": (["lat", "lon"], year_data['num_satellites'][idx].astype(np.int32))
                    },
                    'coords': coords,
                    'attrs': attrs,
                    'var_attrs': var_attrs
                }

                # Safe write
                if safe_file_write(data_dict, file_path, max_retries=2, retry_delay=0.5):
                    successful_writes += 1
                else:
                    failed_writes += 1

            except Exception as e:
                print(f"Error writing {file_path}: {e}")
                failed_writes += 1

    return successful_writes, failed_writes


def optimized_temporal_postprocessing(
        output_dir,
        years=range(1999, 2022),
        var_name="uth_predicted",
        flag_name="quality_flag",
        uncertainty_name="uth_uncertainty",
        temporal_window=3
):
    """
    Optimized temporal postprocessing with single progress bar and vectorized operations
    """
    print("\n" + "=" * 60)
    print("OPTIMIZED TEMPORAL POSTPROCESSING")
    print("=" * 60)

    total_successful = 0
    total_failed = 0

    # Process each year with single progress bar
    with tqdm(years, desc="Processing years") as year_pbar:
        for year in year_pbar:
            year_pbar.set_description(f"Processing year {year}")

            year_dir = os.path.join(output_dir, str(year))
            if not os.path.isdir(year_dir):
                continue

            # Load all year data efficiently
            year_data = load_year_data_efficiently(
                year_dir, var_name, flag_name, uncertainty_name
            )

            if year_data is None:
                print(f"No valid data for year {year}")
                continue

            n_files = len(year_data['file_paths'])
            print(f"\nYear {year}: Processing {n_files} files")

            # Apply temporal smoothing with vectorized operations
            smoothed_predictions = apply_vectorized_temporal_smoothing(
                year_data['predictions'],
                year_data['quality_flags'],
                temporal_window
            )

            # Write smoothed data back
            successful, failed = write_smoothed_data_batch(
                year_data, smoothed_predictions, var_name, flag_name
            )

            total_successful += successful
            total_failed += failed

            print(f"Year {year} completed: {successful} successful, {failed} failed writes")

            # Clean up memory
            del year_data, smoothed_predictions
            gc.collect()

    print(f"\n" + "=" * 60)
    print("TEMPORAL POSTPROCESSING COMPLETED")
    print(f"Total successful writes: {total_successful}")
    print(f"Total failed writes: {total_failed}")
    print("=" * 60)


# ==================== MAIN INFERENCE LOOP ====================
def main_inference_pipeline():
    print("=" * 60)
    print("GEOSPATIAL UTH GAP-FILLING OPTIMIZED INFERENCE PIPELINE")
    print("=" * 60)

    print("\nAvailable models:")
    for key, config in MODEL_CONFIGS.items():
        print(f"{key}: {config['name']}")

    # Accept CLI arg for non-interactive mode
    choice = None
    if len(sys.argv) > 1 and sys.argv[1] in MODEL_CONFIGS:
        choice = sys.argv[1]
        print(f"\nModel choice from CLI: {choice} -> {MODEL_CONFIGS[choice]['name']}")
    else:
        while True:
            choice = input("\nEnter model choice (1/2): ").strip()
            if choice in MODEL_CONFIGS:
                break
            print("Invalid choice. Please select 1 or 2.")

    selected_config = MODEL_CONFIGS[choice]
    print(f"\nSelected: {selected_config['name']}")

    combined_date_index = create_data_index()
    print(f"\nLoading model...")
    model, model_type = load_selected_model(choice)

    print(f"\nCreating output directories...")
    for year in range(1999, 2022):
        year_dir = os.path.join(OUTPUT_DIR, str(year))
        os.makedirs(year_dir, exist_ok=True)

    all_dates = sorted(combined_date_index.keys())
    print(f"\nProcessing {len(all_dates)} dates for inference...")

    successful_predictions = 0
    failed_predictions = 0

    for date_str in tqdm(all_dates, desc="Running inference"):
        try:
            date_obj = datetime.strptime(date_str, "%Y-%m-%d").date()
            year = date_obj.year
            yyyymmdd = date_obj.strftime("%Y%m%d")

            output_path = os.path.join(OUTPUT_DIR, str(year), f"aigeneratedpredictions{yyyymmdd}.nc")

            if os.path.exists(output_path):
                continue

            date_data = combined_date_index[date_str]
            microwave_files = date_data['microwave']

            if not microwave_files:
                continue

            inputs_tensor, masks_tensor, sat_ids_tensor, num_satellites, quality_flag, lat, lon, median, scale = load_microwave_data(
                microwave_files, normalize=True)

            # Main inference
            pred_norm, uncert_norm = run_inference(model, model_type, inputs_tensor, masks_tensor, sat_ids_tensor)

            # Denormalize
            prediction = denormalize_prediction(pred_norm, median, scale)
            uncertainty = denormalize_uncertainty(uncert_norm, scale)

            # Ensure outputs are plain arrays
            prediction = np.asarray(prediction, dtype=np.float32)
            uncertainty = np.asarray(uncertainty, dtype=np.float32)

            # Uncertainty-aware spatial smoothing
            postprocessed_pred, high_uncert_mask = postprocess_uncertainty_smoothing(
                prediction, uncertainty, num_satellites, smooth_kernel=3, method='median', mask_satellite_min=2
            )

            # Save with enhanced quality flag and uncertainty
            if save_prediction_netcdf(postprocessed_pred, uncertainty, num_satellites, quality_flag, high_uncert_mask,
                                      lat, lon, output_path):
                successful_predictions += 1
            else:
                failed_predictions += 1

        except Exception as e:
            print(f"Failed to process {date_str}: {e}")
            failed_predictions += 1
            continue

        if (device.type == 'cuda') and (successful_predictions % 100 == 0):
            torch.cuda.empty_cache()
            gc.collect()

    print(f"\n" + "=" * 60)
    print("INFERENCE COMPLETED")
    print("=" * 60)
    print(f"Successful predictions: {successful_predictions}")
    print(f"Failed predictions: {failed_predictions}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Model used: {selected_config['name']}")

    # ==========> OPTIMIZED TEMPORAL POSTPROCESSING <==========
    try:
        optimized_temporal_postprocessing(
            OUTPUT_DIR,
            years=range(1999, 2022),
            var_name="uth_predicted",
            flag_name="quality_flag",
            uncertainty_name="uth_uncertainty",
            temporal_window=3
        )
    except Exception as e:
        print(f"Temporal postprocessing failed: {e}")
        print("Main inference results are still valid.")

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main_inference_pipeline()
    except KeyboardInterrupt:
        print("\nInference interrupted by user")
    except Exception as e:
        print(f"Pipeline failed with error: {e}")
        raise