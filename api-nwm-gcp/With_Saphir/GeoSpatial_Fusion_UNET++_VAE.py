# corrected_unetpp_vae_saphir_optimal.py

import os
import glob
import gc
import pickle
from collections import defaultdict
from datetime import datetime
import numpy as np
import xarray as xr
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import GradScaler
from torch.amp import autocast
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Constants
LOG2PI = float(np.log(2.0 * np.pi))

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------
# User configuration
# ---------------------------
BASE_DIR = r"C:\S\Programming\NRSC\PythonProject\MICROWAVE_UTH_DATA_NOAA"
SAPHIR_BASE_DIR = r"S:\DATA\SAPHIR_RH_DATA_PROCESSED_2"
CKPT_DIR = "./checkpoints_unetpp_vae_saphir_optimal"
os.makedirs(CKPT_DIR, exist_ok=True)

print(f"Using device: {device}")

# ---------------------------
# Enhanced File Indexing with Saphir Integration
# ---------------------------
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

# Index microwave UTH data
date_file_index = defaultdict(dict)
print("Indexing microwave satellite data files...")
base_dir = BASE_DIR

for sat, folder in satellite_dirs.items():
    folder_path = os.path.join(base_dir, folder)
    if not os.path.isdir(folder_path):
        print(f"Warning: Path does not exist: {folder_path}")
        continue
    nc_files = glob.glob(os.path.join(folder_path, "*.nc"))
    print(f"{sat}: Found {len(nc_files)} files")
    for file_path in nc_files:
        filename = os.path.basename(file_path)
        try:
            date_str = filename[5:13]  # YYYYMMDD
            date_obj = datetime.strptime(date_str, "%Y%m%d").date()
            date_file_index[str(date_obj)][sat] = file_path
        except Exception as e:
            print(f"Failed to parse date from {filename}: {e}")

# Index Saphir RH data
saphir_file_index = {}
print("Indexing Saphir RH data files...")
for year in range(2012, 2022):
    year_dir = os.path.join(SAPHIR_BASE_DIR, str(year))
    if os.path.exists(year_dir):
        nc_files = glob.glob(os.path.join(year_dir, "uthsaphirrh*.nc"))
        for file_path in nc_files:
            filename = os.path.basename(file_path)
            try:
                # Extract date from uthsaphirrh[YYYYMMDD].nc
                date_str = filename[11:19]  # uthsaphirrh + 11 chars = YYYYMMDD
                date = datetime.strptime(date_str, "%Y%m%d").date()
                saphir_file_index[str(date)] = file_path
            except Exception:
                continue

print(f"Indexed {len(date_file_index)} unique microwave dates.")
print(f"Indexed {len(saphir_file_index)} unique Saphir dates.")

# Create combined date index
combined_date_index = {}
for date_str in date_file_index.keys():
    combined_date_index[date_str] = {
        'microwave': date_file_index[date_str],
        'saphir': saphir_file_index.get(date_str, None)
    }

print(f"Combined index created for {len(combined_date_index)} dates.")
saphir_available_dates = sum(1 for v in combined_date_index.values() if v['saphir'] is not None)
print(f"Dates with Saphir data available: {saphir_available_dates}")


# ---------------------------
# Optimal Dataset with Consistent Weighted Saphir Integration
# ---------------------------
class UTHDatasetWithOptimalSaphir(Dataset):
    def __init__(self, dates, combined_date_index, normalize=True, cache_size=100):
        self.dates = dates
        self.combined_date_index = combined_date_index
        self.normalize = normalize
        self._cache = {}
        self._access = {}
        self._cache_size = cache_size

        # Optimized quality-based weights for different data sources
        self.satellite_weights = {
            'NOAA_15': 0.7, 'NOAA_16': 0.8, 'NOAA_17': 0.8,  # Older satellites
            'Metop_A': 1.0, 'Metop_B': 1.0, 'Metop_C': 1.0,  # Newer satellites
            'NOAA_18': 0.9, 'NOAA_19': 0.95
        }
        self.saphir_weight = 2.5  # Increased weight for superior Saphir data

        self.satellite_to_id = {
            'NOAA_15': 0, 'NOAA_16': 1, 'NOAA_17': 2, 'Metop_A': 3,
            'Metop_B': 4, 'Metop_C': 5, 'NOAA_18': 6, 'NOAA_19': 7
        }

        # Template shapes (adjust if different)
        self.template_input = torch.zeros(1, 180, 360, dtype=torch.float32)
        self.template_mask = torch.zeros(180, 360, dtype=torch.float32)

    def __len__(self):
        return len(self.dates)

    def _evict_lru(self):
        if len(self._cache) <= self._cache_size:
            return
        lru_key = min(self._access, key=self._access.get)
        self._cache.pop(lru_key, None)
        self._access.pop(lru_key, None)

    def _load_nc(self, file_path):
        if file_path in self._cache:
            self._access[file_path] += 1
            return self._cache[file_path]

        try:
            with xr.open_dataset(file_path) as ds:
                arr = ds['uth_mean_ascend_descend'].values.astype(np.float32)
                if arr.ndim == 3:
                    arr = arr[0]
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return None

        self._cache[file_path] = arr
        self._access[file_path] = 1
        self._evict_lru()
        return arr

    def load_saphir_data(self, saphir_file_path):
        """Load Saphir RH data and return data with coordinates"""
        try:
            ds = xr.open_dataset(saphir_file_path)
            rh = ds['rh'].values.astype(np.float32)
            latitude = ds['lat'].values.astype(np.float32)
            longitude = ds['lon'].values.astype(np.float32)
            ds.close()
            return rh, latitude, longitude
        except Exception as e:
            print(f"Error loading Saphir data from {saphir_file_path}: {e}")
            return None, None, None

    def _normalize_data(self, data, valid_mask):
        """Consistent normalization function for both inputs and targets"""
        if not self.normalize:
            return data

        if valid_mask.sum() > 100:
            v = data[valid_mask]
            med = np.median(v)
            iqr = np.percentile(v, 75) - np.percentile(v, 25)
            scale = max(iqr, 1e-6)
            return (data - med) / scale
        else:
            return data / 100.0

    def create_optimal_weighted_fusion_target(self, microwave_data_dict, saphir_data, saphir_lat, saphir_lon,
                                              target_shape):
        """Create consistently normalized quality-weighted fusion target"""

        # Initialize fusion arrays
        weighted_sum = np.zeros(target_shape, dtype=np.float32)
        weight_sum = np.zeros(target_shape, dtype=np.float32)
        microwave_mask = np.zeros(target_shape, dtype=np.float32)
        saphir_mask = np.zeros(target_shape, dtype=np.float32)

        # Store raw data for consistent normalization
        all_raw_data = []
        all_weights = []
        all_valid_masks = []

        # Process microwave data with quality weights
        for sat_name, uth_data in microwave_data_dict.items():
            valid_mask = ~np.isnan(uth_data)
            sat_weight = self.satellite_weights.get(sat_name, 1.0)

            all_raw_data.append(uth_data)
            all_weights.append(sat_weight)
            all_valid_masks.append(valid_mask)
            microwave_mask[valid_mask] = 1.0

        # Process Saphir data if available (priority in Â±30Â° band)
        if saphir_data is not None and saphir_data.shape == target_shape:
            # Create Â±30Â° latitude mask
            lat_mask_30 = np.abs(saphir_lat) <= 30
            saphir_valid_mask = ~np.isnan(saphir_data) & lat_mask_30

            all_raw_data.append(saphir_data)
            all_weights.append(self.saphir_weight)
            all_valid_masks.append(saphir_valid_mask)
            saphir_mask[saphir_valid_mask] = 1.0

            # Remove microwave mask where Saphir dominates
            microwave_mask[saphir_valid_mask] = 0.0

        # Create weighted fusion with consistent normalization
        for i, (data, weight, valid_mask) in enumerate(zip(all_raw_data, all_weights, all_valid_masks)):
            # Normalize each dataset consistently
            normalized_data = self._normalize_data(data, valid_mask)

            # Add weighted contribution
            weighted_sum[valid_mask] += normalized_data[valid_mask] * weight
            weight_sum[valid_mask] += weight

        # Compute final weighted fusion
        fused_target = np.zeros_like(weighted_sum)
        valid_regions = weight_sum > 0
        fused_target[valid_regions] = weighted_sum[valid_regions] / weight_sum[valid_regions]

        # Combined mask
        combined_mask = (weight_sum > 0).astype(np.float32)

        return fused_target, combined_mask, microwave_mask, saphir_mask

    def __getitem__(self, idx):
        date = self.dates[idx]
        date_data = self.combined_date_index[date]
        microwave_files = date_data['microwave']
        saphir_file = date_data['saphir']

        inputs = []
        masks = []
        sat_ids = []
        microwave_data_dict = {}

        # Load microwave UTH data
        for sat_name, path in microwave_files.items():
            arr = self._load_nc(path)
            if arr is None:
                continue

            microwave_data_dict[sat_name] = arr
            valid_mask = ~np.isnan(arr)
            arr_filled = np.nan_to_num(arr, nan=0.0).astype(np.float32)

            # Apply consistent normalization
            arr_norm = self._normalize_data(arr_filled, valid_mask)

            inputs.append(torch.from_numpy(arr_norm).unsqueeze(0))
            masks.append(torch.from_numpy(valid_mask.astype(np.float32)))
            sat_ids.append(self.satellite_to_id.get(sat_name, -1))

        # Load Saphir RH data if available
        saphir_data, saphir_lat, saphir_lon = None, None, None
        if saphir_file is not None:
            saphir_data, saphir_lat, saphir_lon = self.load_saphir_data(saphir_file)

        # Get target shape and create consistently normalized weighted fusion target
        target_shape = list(microwave_data_dict.values())[0].shape if microwave_data_dict else (180, 360)

        fused_target, combined_mask, microwave_mask, saphir_mask = self.create_optimal_weighted_fusion_target(
            microwave_data_dict, saphir_data, saphir_lat, saphir_lon, target_shape
        )

        num_valid = len(inputs)
        if num_valid == 0:
            inputs = [self.template_input.clone()]
            masks = [self.template_mask.clone()]
            sat_ids = [-1]

        # Pad to 8 satellites
        while len(inputs) < 8:
            inputs.append(self.template_input.clone())
            masks.append(self.template_mask.clone())
            sat_ids.append(-1)

        inputs = torch.stack(inputs[:8], dim=0)
        masks = torch.stack(masks[:8], dim=0)
        sat_ids = torch.tensor(sat_ids[:8], dtype=torch.long)

        return {
            'date': date,
            'inputs': inputs,  # [N,1,H,W]
            'valid_masks': masks,  # [N,H,W]
            'satellite_ids': sat_ids,  # [N]
            'num_valid_inputs': num_valid,
            'fused_target': torch.tensor(fused_target, dtype=torch.float32),  # [H,W]
            'combined_mask': torch.tensor(combined_mask, dtype=torch.float32),  # [H,W]
            'microwave_mask': torch.tensor(microwave_mask, dtype=torch.float32),  # [H,W]
            'saphir_mask': torch.tensor(saphir_mask, dtype=torch.float32),  # [H,W]
            'saphir_available': saphir_file is not None
        }


# Create train/val splits using combined index
all_dates = sorted(combined_date_index.keys())
print("Total unique dates:", len(all_dates))
split_idx = int(0.9 * len(all_dates))
train_dates = all_dates[:split_idx]
val_dates = all_dates[split_idx:]
print(f"Train dates: {len(train_dates)}")
print(f"Val dates: {len(val_dates)}")


def collate_fn(batch):
    return {
        'inputs': torch.stack([b['inputs'] for b in batch]),  # [B,N,1,H,W]
        'valid_masks': torch.stack([b['valid_masks'] for b in batch]),  # [B,N,H,W]
        'satellite_ids': torch.stack([b['satellite_ids'] for b in batch]),  # [B,N]
        'date': [b['date'] for b in batch],
        'num_valid_inputs': torch.tensor([b['num_valid_inputs'] for b in batch], dtype=torch.long),
        'fused_target': torch.stack([b['fused_target'] for b in batch]),  # [B,H,W]
        'combined_mask': torch.stack([b['combined_mask'] for b in batch]),  # [B,H,W]
        'microwave_mask': torch.stack([b['microwave_mask'] for b in batch]),  # [B,H,W]
        'saphir_mask': torch.stack([b['saphir_mask'] for b in batch]),  # [B,H,W]
        'saphir_available': torch.tensor([b['saphir_available'] for b in batch])
    }


train_ds = UTHDatasetWithOptimalSaphir(train_dates, combined_date_index, normalize=True, cache_size=150)
val_ds = UTHDatasetWithOptimalSaphir(val_dates, combined_date_index, normalize=True, cache_size=50)

train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=0,
                          pin_memory=True, collate_fn=collate_fn, drop_last=True)
val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0,
                        pin_memory=True, collate_fn=collate_fn, drop_last=False)


# ---------------------------
# Model building blocks (preserved from original)
# ---------------------------
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
                if sid >= 0:
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
    """VAE bottleneck with reparameterization trick"""

    def __init__(self, in_channels, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim

        # Encoder to latent parameters
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

        # Decoder from latent
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
        h = h.view(B, -1, 4, 4)  # Reshape to spatial
        h = self.upsample(h)
        # Interpolate to target size
        h = F.interpolate(h, size=target_size, mode='bilinear', align_corners=False)
        return h

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstructed = self.decode(z, x.shape[-2:])
        return reconstructed, mu, logvar


class DenseBlock(nn.Module):
    """Dense connection block for UNet++"""

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
        # Concatenate all inputs
        x = torch.cat(x_list, dim=1)
        out = self.conv(x)
        return self.se(out)


# ---------------------------
# UNet++ Encoder (preserved structure)
# ---------------------------
class UNetPlusEncoderFixed(nn.Module):
    def __init__(self, in_channels=1, base=64, dropout=0.1):
        super().__init__()
        self.base = base

        # Level 0 (input level)
        self.conv00 = ResidualConvBlock(in_channels, base, dropout=dropout)
        self.conv10 = ResidualConvBlock(base, base * 2, dropout=dropout)
        self.conv20 = ResidualConvBlock(base * 2, base * 4, dropout=dropout)
        self.conv30 = ResidualConvBlock(base * 4, base * 8, dropout=dropout)
        self.conv40 = ResidualConvBlock(base * 8, base * 16, dropout=dropout)

        # Level 1 (first nested level)
        self.conv01 = DenseBlock([base, base * 2], base, dropout)
        self.conv11 = DenseBlock([base * 2, base * 4], base * 2, dropout)
        self.conv21 = DenseBlock([base * 4, base * 8], base * 4, dropout)
        self.conv31 = DenseBlock([base * 8, base * 16], base * 8, dropout)

        # Level 2 (second nested level)
        self.conv02 = DenseBlock([base, base, base * 2], base, dropout)
        self.conv12 = DenseBlock([base * 2, base * 2, base * 4], base * 2, dropout)
        self.conv22 = DenseBlock([base * 4, base * 4, base * 8], base * 4, dropout)

        # Level 3 (third nested level)
        self.conv03 = DenseBlock([base, base, base, base * 2], base, dropout)
        self.conv13 = DenseBlock([base * 2, base * 2, base * 2, base * 4], base * 2, dropout)

        # Level 4 (fourth nested level)
        self.conv04 = DenseBlock([base, base, base, base, base * 2], base, dropout)

        self.pool = nn.MaxPool2d(2, 2)

        # VAE bottleneck at the deepest level
        self.vae_bottleneck = VariationalBottleneck(base * 16, latent_dim=512)

    def _align_and_upsample(self, x, target_size):
        """Helper function to properly align tensor sizes"""
        if x.size(2) != target_size[0] or x.size(3) != target_size[1]:
            x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
        return x

    def forward(self, x):
        # Store all intermediate features for nested connections
        features = {}

        # Downsampling path with nested skip connections
        # Level 0,0
        x00 = self.conv00(x)
        features['x00'] = x00

        # Level 1,0
        x10 = self.conv10(self.pool(x00))
        features['x10'] = x10

        # Level 0,1 (first nested connection)
        x10_up = self._align_and_upsample(x10, (x00.size(2), x00.size(3)))
        x01 = self.conv01([x00, x10_up])
        features['x01'] = x01

        # Level 2,0
        x20 = self.conv20(self.pool(x10))
        features['x20'] = x20

        # Level 1,1
        x20_up = self._align_and_upsample(x20, (x10.size(2), x10.size(3)))
        x11 = self.conv11([x10, x20_up])
        features['x11'] = x11

        # Level 0,2
        x11_up = self._align_and_upsample(x11, (x00.size(2), x00.size(3)))
        x02 = self.conv02([x00, x01, x11_up])
        features['x02'] = x02

        # Level 3,0
        x30 = self.conv30(self.pool(x20))
        features['x30'] = x30

        # Level 2,1
        x30_up = self._align_and_upsample(x30, (x20.size(2), x20.size(3)))
        x21 = self.conv21([x20, x30_up])
        features['x21'] = x21

        # Level 1,2
        x21_up = self._align_and_upsample(x21, (x10.size(2), x10.size(3)))
        x12 = self.conv12([x10, x11, x21_up])
        features['x12'] = x12

        # Level 0,3
        x12_up = self._align_and_upsample(x12, (x00.size(2), x00.size(3)))
        x03 = self.conv03([x00, x01, x02, x12_up])
        features['x03'] = x03

        # Level 4,0 (bottleneck)
        x40 = self.conv40(self.pool(x30))
        features['x40'] = x40

        # Apply VAE bottleneck
        x40_vae, mu, logvar = self.vae_bottleneck(x40)
        features['x40_vae'] = x40_vae
        features['mu'] = mu
        features['logvar'] = logvar

        # Level 3,1
        x40_vae_up = self._align_and_upsample(x40_vae, (x30.size(2), x30.size(3)))
        x31 = self.conv31([x30, x40_vae_up])
        features['x31'] = x31

        # Level 2,2
        x31_up = self._align_and_upsample(x31, (x20.size(2), x20.size(3)))
        x22 = self.conv22([x20, x21, x31_up])
        features['x22'] = x22

        # Level 1,3
        x22_up = self._align_and_upsample(x22, (x10.size(2), x10.size(3)))
        x13 = self.conv13([x10, x11, x12, x22_up])
        features['x13'] = x13

        # Level 0,4 (final nested level)
        x13_up = self._align_and_upsample(x13, (x00.size(2), x00.size(3)))
        x04 = self.conv04([x00, x01, x02, x03, x13_up])
        features['x04'] = x04

        return features


# ---------------------------
# Fusion & UNet++ VAE model (preserved structure)
# ---------------------------
class AdvancedWeightedFusion(nn.Module):
    """Advanced fusion with attention and uncertainty weighting"""

    def __init__(self, feature_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(feature_dim, max(feature_dim // 4, 1), 1),
            nn.GELU(),
            nn.Conv2d(max(feature_dim // 4, 1), 1, 1),
            nn.Sigmoid()
        )

    def forward(self, feats, masks):
        # feats: [B,N,C,H,W], masks: [B,N,H,W]
        B, N, C, H, W = feats.shape

        # Compute attention weights for each satellite
        attention_weights = []
        for i in range(N):
            att = self.attention(feats[:, i])  # [B,1,H,W]
            attention_weights.append(att)
        attention_weights = torch.stack(attention_weights, dim=1)  # [B,N,1,H,W]

        # Combine with mask-based weights
        m = masks.unsqueeze(2)  # [B,N,1,H,W]
        pixel_counts = m.sum(dim=(3, 4), keepdim=True).clamp(min=1e-6)  # [B,N,1,1,1]
        reliability_weights = pixel_counts / pixel_counts.sum(dim=1, keepdim=True).clamp(min=1e-6)

        # Final weights combine attention and reliability
        final_weights = attention_weights * reliability_weights * m
        final_weights = final_weights / final_weights.sum(dim=1, keepdim=True).clamp(min=1e-6)

        # Weighted fusion
        fused = (feats * final_weights).sum(dim=1)  # [B,C,H,W]
        return fused


class UNetPlusVAEFixed(nn.Module):
    def __init__(self, in_channels=1, base=64, out_channels=1, num_sat=8, dropout=0.1):
        super().__init__()
        self.base = base
        self.num_sat = num_sat

        # Satellite-specific normalization
        self.sat_norm = SatelliteSpecificNormalization(num_sat, in_channels)

        # Fixed UNet++ encoder
        self.encoder = UNetPlusEncoderFixed(in_channels, base, dropout)

        # Multi-scale fusion modules
        self.fusion_modules = nn.ModuleDict({
            'x04': AdvancedWeightedFusion(base),
            'x03': AdvancedWeightedFusion(base),
            'x02': AdvancedWeightedFusion(base),
            'x01': AdvancedWeightedFusion(base),
            'x00': AdvancedWeightedFusion(base)
        })

        # Final prediction head with uncertainty estimation
        self.final_conv = nn.Sequential(
            nn.Conv2d(base, max(base // 2, 1), 3, padding=1),
            nn.BatchNorm2d(max(base // 2, 1)),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(max(base // 2, 1), out_channels, 1)
        )

        # Uncertainty head (predicts log variance)
        self.uncertainty_head = nn.Sequential(
            nn.Conv2d(base, max(base // 2, 1), 3, padding=1),
            nn.BatchNorm2d(max(base // 2, 1)),
            nn.GELU(),
            nn.Conv2d(max(base // 2, 1), out_channels, 1)
        )

    def forward(self, x, sat_ids, valid_masks):
        # x: [B,8,1,H,W], valid_masks: [B,8,H,W]
        B, N, C, H, W = x.shape

        # Apply satellite-specific normalization
        x = self.sat_norm(x, sat_ids)

        # Encode each satellite separately and collect features at each nested level
        all_features = {level: [] for level in ['x04', 'x03', 'x02', 'x01', 'x00']}
        mu_list = []
        logvar_list = []

        for i in range(N):
            if valid_masks[:, i].sum() == 0:
                # Create zero features for invalid satellites
                for level in all_features.keys():
                    all_features[level].append(torch.zeros(B, self.base, H, W,
                                                           device=x.device, dtype=x.dtype))
                mu_list.append(torch.zeros(B, 512, device=x.device, dtype=x.dtype))
                logvar_list.append(torch.zeros(B, 512, device=x.device, dtype=x.dtype))
            else:
                # Encode valid satellite
                features = self.encoder(x[:, i])
                for level in all_features.keys():
                    all_features[level].append(features[level])
                mu_list.append(features['mu'])
                logvar_list.append(features['logvar'])

        # Stack features across satellites
        for level in all_features.keys():
            all_features[level] = torch.stack(all_features[level], dim=1)  # [B,N,C,H,W]

        # Aggregate VAE parameters
        mu_agg = torch.stack(mu_list, dim=1).mean(dim=1)  # [B, 512]
        logvar_agg = torch.stack(logvar_list, dim=1).mean(dim=1)  # [B, 512]

        # Fuse features at the finest level (x04)
        fused_features = self.fusion_modules['x04'](all_features['x04'], valid_masks)

        # Generate predictions
        prediction = self.final_conv(fused_features)
        uncertainty = self.uncertainty_head(fused_features)

        return {
            'prediction': prediction,
            'uncertainty': uncertainty,
            'mu': mu_agg,
            'logvar': logvar_agg,
            'features': all_features
        }


# ---------------------------
# Optimal Saphir-Aware Hybrid VAE Loss (All Issues Fixed)
# ---------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from pytorch_msssim import ms_ssim  # pip install pytorch-msssim
except ImportError:
    print("Warning: pytorch-msssim not available. SSIM loss will be disabled.")
    ms_ssim = None


class OptimalSaphirAwareVAELoss(nn.Module):
    """
    Optimized hybrid VAE loss with Saphir-aware weighting and proper scaling:
    - Balanced weighted MSE (stabilized scaling)
    - Proper SSIM handling (disabled if unavailable)
    - KL divergence with annealing
    - TV regularization (minimal weight)
    - Saphir-aware weighting with normalized scaling
    """

    def __init__(self,
                 alpha=0.8,  # MSE weight (increased)
                 beta=0.1,  # SSIM weight (reduced)
                 gamma_max=0.005,  # KL weight (reduced)
                 tv_weight=1e-4,  # TV weight (reduced)
                 microwave_weight=1.0,
                 saphir_weight=1.5,  # Reduced from 2.0 for stability
                 kl_warmup_epochs=10):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma_max = gamma_max
        self.tv_weight = tv_weight
        self.microwave_weight = microwave_weight
        self.saphir_weight = saphir_weight
        self.kl_warmup_epochs = max(1, kl_warmup_epochs)

    def _fix_mask_dims(self, mask):
        """Fix mask dimensions to be compatible with 4D tensors"""
        if mask.ndim == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)  # [H,W] -> [1,1,H,W]
        elif mask.ndim == 3:
            mask = mask.unsqueeze(1)  # [B,H,W] -> [B,1,H,W]
        return mask

    def _masked_mse(self, pred, target, mask):
        """Masked MSE loss with proper dimension handling"""
        mask = self._fix_mask_dims(mask)
        diff2 = (pred - target) ** 2
        denom = mask.sum().clamp(min=1.0).to(pred.device)
        return (diff2 * mask).sum() / denom

    def _masked_ssim_loss(self, pred, target, mask):
        """Proper SSIM loss implementation with fallback"""
        if ms_ssim is None:
            # Disable SSIM if library not available
            return torch.tensor(0.0, device=pred.device)

        mask = self._fix_mask_dims(mask)

        try:
            # Only compute SSIM if most pixels are valid
            if mask.float().mean() > 0.7:
                return 1.0 - ms_ssim(pred, target, data_range=6.0, size_average=True)
            else:
                # Fallback to zero if too many masked pixels
                return torch.tensor(0.0, device=pred.device)
        except Exception:
            # Fallback on any error
            return torch.tensor(0.0, device=pred.device)

    def _tv_loss(self, pred, mask):
        """Total variation loss for spatial smoothness"""
        mask = self._fix_mask_dims(mask)

        dx = torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1])
        dy = torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :])
        mx = mask[:, :, :, 1:] * mask[:, :, :, :-1]
        my = mask[:, :, 1:, :] * mask[:, :, :-1, :]

        denom_x = mx.sum().clamp(min=1.0).to(pred.device)
        denom_y = my.sum().clamp(min=1.0).to(pred.device)

        return ((dx * mx).sum() / denom_x + (dy * my).sum() / denom_y) * 0.5

    def _kl_div(self, mu, logvar):
        """KL divergence loss"""
        val = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        return val.mean()

    def saphir_aware_reconstruction_loss(self, pred, target, microwave_mask, saphir_mask):
        """Compute Saphir-aware weighted reconstruction loss with proper scaling"""

        total_mse = torch.tensor(0.0, device=pred.device)
        total_ssim = torch.tensor(0.0, device=pred.device)
        total_weight = torch.tensor(0.0, device=pred.device)

        # MSE losses for different regions
        if microwave_mask.sum() > 0:
            microwave_mse = self._masked_mse(pred, target, microwave_mask)
            total_mse += microwave_mse * self.microwave_weight
            total_weight += self.microwave_weight

        if saphir_mask.sum() > 0:
            saphir_mse = self._masked_mse(pred, target, saphir_mask)
            total_mse += saphir_mse * self.saphir_weight
            total_weight += self.saphir_weight

        # SSIM losses for different regions
        if self.beta > 0:  # Only compute if SSIM weight > 0
            if microwave_mask.sum() > 0:
                microwave_ssim = self._masked_ssim_loss(pred, target, microwave_mask)
                total_ssim += microwave_ssim * self.microwave_weight

            if saphir_mask.sum() > 0:
                saphir_ssim = self._masked_ssim_loss(pred, target, saphir_mask)
                total_ssim += saphir_ssim * self.saphir_weight

        # Normalize by total weight
        if total_weight > 0:
            total_mse = total_mse / total_weight
            if self.beta > 0:
                total_ssim = total_ssim / total_weight

        return total_mse, total_ssim

    def forward(self, model_output, fused_target, microwave_mask, saphir_mask, epoch=0):
        """
        Optimized forward pass with proper scaling and error handling
        """
        pred = model_output['prediction']
        mu = model_output.get('mu', None)
        logvar_z = model_output.get('logvar', None)

        # Ensure target has channel dimension
        if fused_target.ndim == 3:
            fused_target = fused_target.unsqueeze(1)  # [B,H,W] -> [B,1,H,W]

        # Saphir-aware reconstruction losses
        mse_loss, ssim_loss = self.saphir_aware_reconstruction_loss(
            pred, fused_target, microwave_mask, saphir_mask
        )

        # Combined mask for TV loss
        combined_mask = (microwave_mask + saphir_mask).clamp(max=1.0)
        tv_loss = self._tv_loss(pred, combined_mask)

        # KL divergence with annealing
        kl_loss = torch.tensor(0.0, device=pred.device)
        if mu is not None and logvar_z is not None:
            kl_weight = self.gamma_max * min(1.0, epoch / self.kl_warmup_epochs)
            kl_loss = kl_weight * self._kl_div(mu, logvar_z)

        # Total loss with proper scaling
        total = (self.alpha * mse_loss +
                 self.beta * ssim_loss +
                 kl_loss +
                 self.tv_weight * tv_loss)

        # Clamp to prevent exploding loss
        total = torch.clamp(total, max=100.0)

        metrics = {
            'total': float(total.item()),
            'mse': float(mse_loss.item()),
            'ssim': float(ssim_loss.item()) if hasattr(ssim_loss, 'item') else float(ssim_loss),
            'kl': float(kl_loss.item()) if hasattr(kl_loss, 'item') else float(kl_loss),
            'tv': float(tv_loss.item())
        }

        return total, metrics


# ---------------------------
# Training utilities with gradient clipping
# ---------------------------
EPOCHS = 25
LEARNING_RATE = 5e-5  # Reduced learning rate for stability
SAVE_EVERY = 5

# Initialize model, optimizer, scheduler, criterion
model = UNetPlusVAEFixed(in_channels=1, base=64, out_channels=1, num_sat=8, dropout=0.1).to(device)
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=7)
criterion = OptimalSaphirAwareVAELoss().to(device)

# GradScaler
scaler = GradScaler()


def train_one_epoch(model, dataloader, optimizer, criterion, device, epoch, scaler):
    model.train()
    total_loss = 0.0
    total_metrics = defaultdict(float)
    saphir_samples = 0
    total_samples = 0

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")

    for batch_idx, batch in enumerate(progress_bar):
        inputs = batch['inputs'].to(device)  # [B,N,1,H,W]
        valid_masks = batch['valid_masks'].to(device)  # [B,N,H,W]
        satellite_ids = batch['satellite_ids'].to(device)
        fused_target = batch['fused_target'].to(device)  # [B,H,W]
        microwave_mask = batch['microwave_mask'].to(device)  # [B,H,W]
        saphir_mask = batch['saphir_mask'].to(device)  # [B,H,W]
        saphir_available = batch['saphir_available']

        optimizer.zero_grad()

        with autocast(device_type='cuda' if device.type == 'cuda' else 'cpu'):
            output = model(inputs, satellite_ids, valid_masks)
            loss, metrics = criterion(output, fused_target, microwave_mask, saphir_mask, epoch=epoch)

        # Gradient clipping for stability
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        for key, value in metrics.items():
            total_metrics[key] += value

        # Track Saphir usage
        total_samples += len(saphir_available)
        saphir_samples += saphir_available.sum().item()

        progress_bar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'mse': f"{metrics.get('mse', 0.0):.4f}",
            'ssim': f"{metrics.get('ssim', 0.0):.4f}",
            'kl': f"{metrics.get('kl', 0.0):.6f}",
            'saphir_pct': f"{100 * saphir_samples / total_samples:.1f}%"
        })

    avg_loss = total_loss / max(1, len(dataloader))
    avg_metrics = {k: v / max(1, len(dataloader)) for k, v in total_metrics.items()}
    avg_metrics['saphir_coverage'] = 100 * saphir_samples / total_samples if total_samples > 0 else 0

    return avg_loss, avg_metrics


def validate_one_epoch(model, dataloader, criterion, device, epoch):
    model.eval()
    total_loss = 0.0
    total_metrics = defaultdict(float)
    saphir_samples = 0
    total_samples = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            inputs = batch['inputs'].to(device)
            valid_masks = batch['valid_masks'].to(device)
            satellite_ids = batch['satellite_ids'].to(device)
            fused_target = batch['fused_target'].to(device)
            microwave_mask = batch['microwave_mask'].to(device)
            saphir_mask = batch['saphir_mask'].to(device)
            saphir_available = batch['saphir_available']

            with torch.amp.autocast(device_type=device.type if isinstance(device, torch.device) else 'cuda'):
                output = model(inputs, satellite_ids, valid_masks)
                loss, metrics = criterion(output, fused_target, microwave_mask, saphir_mask, epoch=epoch)

            total_loss += loss.item()
            for key, value in metrics.items():
                total_metrics[key] += value

            total_samples += len(saphir_available)
            saphir_samples += saphir_available.sum().item()

    avg_loss = total_loss / max(1, len(dataloader))
    avg_metrics = {k: v / max(1, len(dataloader)) for k, v in total_metrics.items()}
    avg_metrics['saphir_coverage'] = 100 * saphir_samples / total_samples if total_samples > 0 else 0

    return avg_loss, avg_metrics


def train_model():
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []

    print("Starting optimal training with Saphir-aware weighted fusion...")
    print(f"Total training samples: {len(train_ds)}")
    print(f"Total validation samples: {len(val_ds)}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    for epoch in range(1, EPOCHS + 1):
        print(f"\nEpoch {epoch}/{EPOCHS}")
        print("-" * 50)

        # Training
        train_loss, train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch, scaler
        )

        # Validation
        val_loss, val_metrics = validate_one_epoch(
            model, val_loader, criterion, device, epoch
        )

        # Learning rate scheduling
        scheduler.step(val_loss)

        # Store losses
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # Print epoch summary
        print(f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
        print(
            f"Train Saphir Coverage: {train_metrics['saphir_coverage']:.1f}% | Val Saphir Coverage: {val_metrics['saphir_coverage']:.1f}%")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.8f}")

        # Track best validation loss and save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'val_metrics': val_metrics,
                'model_config': {
                    'in_channels': 1, 'base': 64, 'out_channels': 1,
                    'num_sat': 8, 'dropout': 0.1
                }
            }
            best_path = os.path.join(CKPT_DIR, "best_model.pth")
            torch.save(best_checkpoint, best_path)
            print(f"âœ“ New best validation loss: {val_loss:.6f} (saved to {best_path})")
        else:
            patience_counter += 1

        # Optional: save periodic checkpoints
        if epoch % SAVE_EVERY == 0:
            path = os.path.join(CKPT_DIR, f"checkpoint_epoch_{epoch}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_losses': train_losses,
                'val_losses': val_losses
            }, path)
            print(f"Saved checkpoint: {path}")

        # Memory cleanup
        torch.cuda.empty_cache()
        gc.collect()

    print(f"\nTraining completed!")
    print(f"Best validation loss achieved: {best_val_loss:.6f}")

    # Save the final model
    print("\nSaving final model...")
    final_model_path_pth = os.path.join(CKPT_DIR, "UNET++_VAE_Saphir_Optimal.pth")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_val_loss': best_val_loss,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'model_config': {
            'in_channels': 1, 'base': 64, 'out_channels': 1,
            'num_sat': 8, 'dropout': 0.1
        }
    }, final_model_path_pth)
    print(f"âœ“ Model saved as: {final_model_path_pth}")

    final_model_path_h5 = os.path.join(CKPT_DIR, "UNET++_VAE_Saphir_Optimal.h5")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_val_loss': best_val_loss,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'model_config': {
            'in_channels': 1, 'base': 64, 'out_channels': 1,
            'num_sat': 8, 'dropout': 0.1
        }
    }, final_model_path_h5)
    print(f"âœ“ Model saved as: {final_model_path_h5}")

    return train_losses, val_losses


# ---------------------------
# Evaluation utilities and metrics (preserved)
# ---------------------------
def calculate_ergas(true, pred, mean_true=None):
    """Calculate ERGAS robustly on valid pixels"""
    mask = ~(np.isnan(true) | np.isnan(pred) | np.isinf(true) | np.isinf(pred))
    if mask.sum() == 0:
        return None

    true_valid = true[mask]
    pred_valid = pred[mask]

    if mean_true is None:
        mean_true = np.mean(true_valid)

    if mean_true == 0:
        return float('inf')

    mse = np.mean((true_valid - pred_valid) ** 2)
    ergas = 100 * np.sqrt(mse) / mean_true
    return ergas


def calculate_metrics(y_true, y_pred, data_range=None):
    """Calculate comprehensive metrics"""
    if y_true.shape != y_pred.shape:
        return None

    mask = ~(np.isnan(y_true) | np.isnan(y_pred) | np.isinf(y_true) | np.isinf(y_pred))
    if mask.sum() == 0:
        return None

    y_true_valid = y_true[mask]
    y_pred_valid = y_pred[mask]

    mse = mean_squared_error(y_true_valid, y_pred_valid)
    rmse = float(np.sqrt(mse))
    mae = mean_absolute_error(y_true_valid, y_pred_valid)

    if data_range is None:
        data_range = np.max(y_true_valid) - np.min(y_true_valid)
    if data_range == 0:
        data_range = 1.0

    # PSNR (manual)
    try:
        psnr = 10.0 * np.log10((data_range ** 2) / (mse + 1e-12))
    except Exception:
        psnr = None

    # ERGAS
    try:
        ergas = calculate_ergas(y_true, y_pred)
    except Exception:
        ergas = None

    # SSIM: only compute if there are no masked pixels (safe case) and shape is 2D
    ssim = None
    try:
        if mask.all() and y_true.ndim == 2:
            from skimage.metrics import structural_similarity as ssim_fn
            ssim = float(ssim_fn(y_true, y_pred, data_range=data_range))
    except Exception:
        ssim = None

    return {
        'MSE': float(mse),
        'RMSE': float(rmse),
        'MAE': float(mae),
        'PSNR': float(psnr) if psnr is not None else None,
        'ERGAS': float(ergas) if ergas is not None else None,
        'SSIM': float(ssim) if ssim is not None else None
    }


def load_best_model():
    """Load the best trained model"""
    best_model_path = os.path.join(CKPT_DIR, "best_model.pth")
    if not os.path.exists(best_model_path):
        raise FileNotFoundError(f"Best model not found at {best_model_path}")

    checkpoint = torch.load(best_model_path, map_location=device)
    model_local = UNetPlusVAEFixed(in_channels=1, base=64, out_channels=1, num_sat=8, dropout=0.1).to(device)
    model_local.load_state_dict(checkpoint['model_state_dict'])

    print(f"Loaded best model from epoch {checkpoint['epoch']}")
    print(f"Best validation loss: {checkpoint.get('best_val_loss', float('nan')):.6f}")

    return model_local


def evaluate_model(model_eval, dataloader, device, num_samples=808):
    """Comprehensive model evaluation"""
    model_eval.eval()
    all_predictions = []
    all_targets = []
    all_metrics = []
    sample_count = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            if num_samples and sample_count >= num_samples:
                break

            inputs = batch['inputs'].to(device)
            valid_masks = batch['valid_masks'].to(device)
            satellite_ids = batch['satellite_ids'].to(device)
            fused_target = batch['fused_target'].to(device)
            combined_mask = batch['combined_mask'].to(device)

            with torch.amp.autocast(device_type=device.type if isinstance(device, torch.device) else 'cuda'):
                output = model_eval(inputs, satellite_ids, valid_masks)

            predictions = output['prediction'].cpu().numpy()  # [B,1,H,W]
            targets = fused_target.cpu().numpy()  # [B,H,W]
            masks = combined_mask.cpu().numpy()  # [B,H,W]

            # Process each sample in batch
            batch_size = predictions.shape[0]
            for i in range(batch_size):
                if sample_count >= num_samples:
                    break

                pred = predictions[i, 0]
                true = targets[i]
                mask = masks[i]

                valid_pixels = mask > 0
                if valid_pixels.sum() > 0:
                    pred_masked = pred * mask
                    true_masked = true * mask

                    metrics = calculate_metrics(true_masked, pred_masked)
                    if metrics:
                        all_metrics.append(metrics)
                        all_predictions.append(pred_masked)
                        all_targets.append(true_masked)

                sample_count += 1

    return all_predictions, all_targets, all_metrics


def print_evaluation_summary(metrics_list):
    """Print summary statistics of evaluation metrics"""
    if not metrics_list:
        print("No valid metrics to summarize")
        return

    aggregated = defaultdict(list)
    for metrics in metrics_list:
        for key, value in metrics.items():
            if value is not None and not np.isnan(value) and not np.isinf(value):
                aggregated[key].append(value)

    print("\n" + "=" * 60)
    print("QUANTITATIVE EVALUATION RESULTS (OPTIMAL SAPHIR-WEIGHTED)")
    print("=" * 60)

    for metric_name, values in aggregated.items():
        if values:
            mean_val = np.mean(values)
            std_val = np.std(values)
            min_val = np.min(values)
            max_val = np.max(values)
            print(f"{metric_name:>8}: {mean_val:8.4f} Â± {std_val:6.4f} "
                  f"(min: {min_val:7.4f}, max: {max_val:7.4f})")
    print("=" * 60)


# Visualization functions
def plot_training_curves(train_losses, val_losses):
    """Plot training and validation loss curves"""
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    plt.title('Training and Validation Loss Curves (Optimal Saphir-Weighted)', fontsize=16)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(CKPT_DIR, 'training_curves_optimal.png'), dpi=300, bbox_inches='tight')
    plt.show()


def visualize_predictions(predictions, targets, num_samples=5):
    """Visualize sample predictions vs targets"""
    fig, axes = plt.subplots(2, num_samples, figsize=(15, 6))
    for i in range(min(num_samples, len(predictions))):
        # Target
        axes[0, i].imshow(targets[i], cmap='viridis', aspect='auto')
        axes[0, i].set_title(f'Target {i + 1}')
        axes[0, i].axis('off')

        # Prediction
        axes[1, i].imshow(predictions[i], cmap='viridis', aspect='auto')
        axes[1, i].set_title(f'Prediction {i + 1}')
        axes[1, i].axis('off')

    plt.suptitle('Optimal Saphir-Weighted VAE Predictions', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(CKPT_DIR, 'sample_predictions_optimal.png'), dpi=300, bbox_inches='tight')
    plt.show()


# ---------------------------
# Run training and evaluation
# ---------------------------
if __name__ == "__main__":
    try:
        train_losses, val_losses = train_model()

        print("\nTraining completed successfully with optimal Saphir-weighted fusion!")
        print(f"Models saved in: {CKPT_DIR}")
        print(" - UNET++_VAE_Saphir_Optimal.pth")
        print(" - UNET++_VAE_Saphir_Optimal.h5")
        print(" - best_model.pth (best validation model)")

        # Evaluate best model if available
        best_path = os.path.join(CKPT_DIR, "best_model.pth")
        if os.path.exists(best_path):
            print("\nLoading best model for evaluation...")
            best_model = load_best_model()

            print("Evaluating on validation set...")
            predictions, targets, metrics_list = evaluate_model(best_model, val_loader, device, num_samples=808)

            print_evaluation_summary(metrics_list)

            # Save evaluation results
            evaluation_results = {
                'predictions': predictions[:10],
                'targets': targets[:10],
                'metrics': metrics_list,
                'summary': {}
            }

            for metric in ['MSE', 'RMSE', 'MAE', 'PSNR', 'ERGAS', 'SSIM']:
                values = [m[metric] for m in metrics_list if (
                            metric in m and m[metric] is not None and not np.isnan(m[metric]) and not np.isinf(
                        m[metric]))]
                if values:
                    evaluation_results['summary'][metric] = {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'count': len(values)
                    }

            results_path = os.path.join(CKPT_DIR, "evaluation_results_optimal.pkl")
            with open(results_path, 'wb') as f:
                pickle.dump(evaluation_results, f)
            print(f"\nEvaluation results saved to: {results_path}")

            # Generate plots
            plot_training_curves(train_losses, val_losses)
            visualize_predictions(predictions, targets)

    except Exception as e:
        print(f" Training failed with error: {e}")
        debug_path = os.path.join(CKPT_DIR, "debug_state.pth")
        try:
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'error': str(e)
            }, debug_path)
            print(f"Debug state saved to: {debug_path}")
        except Exception as save_err:
            print(f"Failed to save debug state: {save_err}")
        raise

    print("\nScript completed successfully with optimal Saphir integration!")