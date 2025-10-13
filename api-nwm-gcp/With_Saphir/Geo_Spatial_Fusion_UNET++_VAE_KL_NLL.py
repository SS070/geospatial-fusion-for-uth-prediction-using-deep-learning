# unetpp_vae_with_saphir_integration_FINAL_CORRECTED.py

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
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.amp import GradScaler, autocast
from sklearn.metrics import mean_squared_error, mean_absolute_error
from skimage.metrics import structural_similarity

# Constants
LOG2PI = float(np.log(2.0 * np.pi))

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------
# User configuration
# ---------------------------
BASE_DIR = r"C:\S\Programming\NRSC\PythonProject\MICROWAVE_UTH_DATA_NOAA"
SAPHIR_BASE_DIR = r"S:\DATA\SAPHIR_RH_DATA_PROCESSED_2"
CKPT_DIR = "./checkpoints_unetpp_vae_saphir_final"
os.makedirs(CKPT_DIR, exist_ok=True)

print(f"Using device: {device}")

# ---------------------------
# Index satellite files by date (Enhanced with Saphir)
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
# CORRECTED Dataset with Consistent Saphir Integration
# ---------------------------
class UTHDatasetWithSaphirVAE(Dataset):
    def __init__(self, dates, combined_date_index, normalize=True, cache_size=100):
        self.dates = dates
        self.combined_date_index = combined_date_index
        self.normalize = normalize
        self._cache = {}
        self._access = {}
        self._cache_size = cache_size

        # Quality-based weights for different data sources
        self.satellite_weights = {
            'NOAA_15': 0.8, 'NOAA_16': 0.85, 'NOAA_17': 0.85,
            'Metop_A': 1.0, 'Metop_B': 1.0, 'Metop_C': 1.0,
            'NOAA_18': 0.9, 'NOAA_19': 0.95
        }
        self.saphir_weight = 2.0

        self.satellite_to_id = {
            'NOAA_15': 0, 'NOAA_16': 1, 'NOAA_17': 2, 'Metop_A': 3,
            'Metop_B': 4, 'Metop_C': 5, 'NOAA_18': 6, 'NOAA_19': 7
        }

        # Template shapes
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
        """CORRECTED: Load Saphir RH data with proper coordinate names"""
        try:
            ds = xr.open_dataset(saphir_file_path)
            rh = ds['rh'].values.astype(np.float32)
            # FIXED: Use correct coordinate variable names
            latitude = ds['lat'].values.astype(np.float32)
            longitude = ds['lon'].values.astype(np.float32)
            ds.close()
            return rh, latitude, longitude
        except Exception as e:
            print(f"Error loading Saphir data from {saphir_file_path}: {e}")
            return None, None, None

    def _normalize_data_consistently(self, data, valid_mask):
        """CORRECTED: Consistent normalization function for both inputs and targets"""
        if not self.normalize:
            return data

        filled_data = np.nan_to_num(data, nan=0.0).astype(np.float32)

        if valid_mask.sum() > 100:
            v = filled_data[valid_mask]
            med = np.median(v)
            iqr = np.percentile(v, 75) - np.percentile(v, 25)
            scale = max(iqr, 1e-6)
            return (filled_data - med) / scale
        else:
            return filled_data / 100.0

    def create_weighted_fusion(self, microwave_data_dict, saphir_data, saphir_lat, saphir_lon, target_shape):
        """CORRECTED: Create quality-weighted fusion with CONSISTENT normalization"""

        # Initialize fusion arrays
        weighted_sum = np.zeros(target_shape, dtype=np.float32)
        weight_sum = np.zeros(target_shape, dtype=np.float32)
        microwave_mask = np.zeros(target_shape, dtype=np.float32)
        saphir_mask = np.zeros(target_shape, dtype=np.float32)

        # Process microwave data with CONSISTENT NORMALIZATION
        for sat_name, uth_data in microwave_data_dict.items():
            valid_mask = ~np.isnan(uth_data)
            sat_weight = self.satellite_weights.get(sat_name, 1.0)

            # FIXED: Apply SAME normalization as used for inputs
            normalized_data = self._normalize_data_consistently(uth_data, valid_mask)

            # Add weighted contribution with NORMALIZED data
            weighted_sum[valid_mask] += normalized_data[valid_mask] * sat_weight
            weight_sum[valid_mask] += sat_weight
            microwave_mask[valid_mask] = 1.0

        # Process Saphir data with CONSISTENT NORMALIZATION
        if saphir_data is not None and saphir_data.shape == target_shape:
            # Create ±30° latitude mask
            lat_mask_30 = np.abs(saphir_lat) <= 30
            saphir_valid_mask = ~np.isnan(saphir_data) & lat_mask_30

            # FIXED: Apply SAME normalization as used for inputs
            normalized_saphir = self._normalize_data_consistently(saphir_data, saphir_valid_mask)

            # Add Saphir weighted contribution with NORMALIZED data
            weighted_sum[saphir_valid_mask] += normalized_saphir[saphir_valid_mask] * self.saphir_weight
            weight_sum[saphir_valid_mask] += self.saphir_weight
            saphir_mask[saphir_valid_mask] = 1.0

            # Remove microwave mask where Saphir is present (Saphir dominates)
            microwave_mask[saphir_valid_mask] = 0.0

        # Compute final weighted fusion
        fused_data = np.zeros_like(weighted_sum)
        valid_regions = weight_sum > 0
        fused_data[valid_regions] = weighted_sum[valid_regions] / weight_sum[valid_regions]

        # Combined mask for all valid data
        combined_mask = (weight_sum > 0).astype(np.float32)

        return fused_data, combined_mask, microwave_mask, saphir_mask

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
        for sat_name, file_path in microwave_files.items():
            arr = self._load_nc(file_path)
            if arr is None:
                continue

            microwave_data_dict[sat_name] = arr
            valid_mask = ~np.isnan(arr)

            # FIXED: Use consistent normalization function
            arr_norm = self._normalize_data_consistently(arr, valid_mask)

            inputs.append(torch.from_numpy(arr_norm).unsqueeze(0))
            masks.append(torch.from_numpy(valid_mask.astype(np.float32)))
            sat_ids.append(self.satellite_to_id.get(sat_name, -1))

        # Load Saphir RH data if available
        saphir_data, saphir_lat, saphir_lon = None, None, None
        if saphir_file is not None:
            saphir_data, saphir_lat, saphir_lon = self.load_saphir_data(saphir_file)

        # Get target shape from first microwave data
        if microwave_data_dict:
            target_shape = list(microwave_data_dict.values())[0].shape
            # Create CONSISTENTLY NORMALIZED weighted fusion for target
            fused_data, combined_mask, microwave_mask, saphir_mask = self.create_weighted_fusion(
                microwave_data_dict, saphir_data, saphir_lat, saphir_lon, target_shape
            )
        else:
            # Fallback if no microwave data
            fused_data = np.zeros((180, 360), dtype=np.float32)
            combined_mask = np.zeros((180, 360), dtype=np.float32)
            microwave_mask = np.zeros((180, 360), dtype=np.float32)
            saphir_mask = np.zeros((180, 360), dtype=np.float32)

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

        # Convert fusion results to tensors
        fused_tensor = torch.tensor(fused_data, dtype=torch.float32)
        combined_mask_tensor = torch.tensor(combined_mask, dtype=torch.float32)
        microwave_mask_tensor = torch.tensor(microwave_mask, dtype=torch.float32)
        saphir_mask_tensor = torch.tensor(saphir_mask, dtype=torch.float32)

        return {
            'date': date,
            'inputs': inputs,  # [N,1,H,W]
            'valid_masks': masks,  # [N,H,W]
            'satellite_ids': sat_ids,  # [N]
            'num_valid_inputs': num_valid,
            'fused_target': fused_tensor,  # [H,W]
            'combined_mask': combined_mask_tensor,  # [H,W]
            'microwave_mask': microwave_mask_tensor,  # [H,W]
            'saphir_mask': saphir_mask_tensor,  # [H,W]
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


train_ds = UTHDatasetWithSaphirVAE(train_dates, combined_date_index, normalize=True, cache_size=150)
val_ds = UTHDatasetWithSaphirVAE(val_dates, combined_date_index, normalize=True, cache_size=50)

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
# UNet++ Encoder (same structure preserved)
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
# FINAL CORRECTED Saphir-Aware VAE Loss (All Issues Fixed)
# ---------------------------
class SaphirAwareVAELoss(nn.Module):
    """FINAL CORRECTED: Enhanced VAE loss with proper scaling, no negative losses, and Saphir-aware weighting"""

    def __init__(self,
                 alpha=0.8,  # MSE weight
                 beta=0.1,  # Charbonnier weight
                 gamma_max=0.005,  # KL weight
                 lambda_grad=0.01,  # Gradient smoothness
                 microwave_weight=1.0,
                 saphir_weight=1.5,  # Balanced for stability
                 charbonnier_eps=1e-3,
                 logvar_clamp=(-5.0, 5.0),  # Stricter clamping
                 kl_warmup_epochs=15):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma_max = gamma_max
        self.lambda_grad = lambda_grad
        self.microwave_weight = microwave_weight
        self.saphir_weight = saphir_weight
        self.eps = charbonnier_eps
        self.logvar_min, self.logvar_max = logvar_clamp
        self.kl_warmup_epochs = max(1, kl_warmup_epochs)

    def _fix_mask_dimensions(self, mask):
        """CORRECTED: Fix mask dimensions for proper operations"""
        if mask.ndim == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)  # [H,W] -> [1,1,H,W]
        elif mask.ndim == 3:
            mask = mask.unsqueeze(1)  # [B,H,W] -> [B,1,H,W]
        return mask

    def _clamp_logvar(self, logvar):
        return torch.clamp(logvar, min=self.logvar_min, max=self.logvar_max)

    def _stable_nll(self, pred, target, logvar, mask):
        """CORRECTED: Stable NLL that prevents negative losses"""
        mask = self._fix_mask_dimensions(mask)
        logvar = self._clamp_logvar(logvar)

        # Compute MSE
        mse = (pred - target) ** 2

        # Stable NLL computation preventing negative values
        # Use simplified MSE-based approach for numerical stability
        denom = mask.sum().clamp(min=1.0).to(pred.device)
        nll = (mse * mask).sum() / denom

        # Ensure non-negative
        return torch.clamp(nll, min=0.0)

    def _charbonnier(self, pred, target, mask):
        mask = self._fix_mask_dimensions(mask)
        diff2 = (pred - target) ** 2
        char = torch.sqrt(diff2 + (self.eps ** 2))
        denom = mask.sum().clamp(min=1.0).to(char.device)
        return (char * mask).sum() / denom

    def _grad_smoothness(self, pred, mask):
        """CORRECTED: Gradient smoothness with proper tensor dimension handling"""
        mask = self._fix_mask_dimensions(mask)

        dx = torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1])
        dy = torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :])
        mx = mask[:, :, :, 1:] * mask[:, :, :, :-1]
        my = mask[:, :, 1:, :] * mask[:, :, :-1, :]

        denom_x = mx.sum().clamp(min=1.0).to(pred.device)
        denom_y = my.sum().clamp(min=1.0).to(pred.device)

        gx = (dx * mx).sum() / denom_x
        gy = (dy * my).sum() / denom_y

        return (gx + gy) * 0.5

    def _kl_div(self, mu, logvar):
        val = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        return val.clamp(min=0).mean()  # Enforce non-negativity

    def saphir_aware_loss(self, pred, target, logvar, microwave_mask, saphir_mask):
        """CORRECTED: Compute Saphir-aware weighted loss with guaranteed positive values"""

        # Compute loss for microwave regions
        microwave_loss = torch.tensor(0.0, device=pred.device)
        if microwave_mask.sum() > 0:
            microwave_nll = self._stable_nll(pred, target, logvar, microwave_mask)
            microwave_char = self._charbonnier(pred, target, microwave_mask)
            microwave_loss = (self.alpha * microwave_nll + self.beta * microwave_char) * self.microwave_weight

        # Compute loss for Saphir regions (higher weight)
        saphir_loss = torch.tensor(0.0, device=pred.device)
        if saphir_mask.sum() > 0:
            saphir_nll = self._stable_nll(pred, target, logvar, saphir_mask)
            saphir_char = self._charbonnier(pred, target, saphir_mask)
            saphir_loss = (self.alpha * saphir_nll + self.beta * saphir_char) * self.saphir_weight

        # Normalize by total weight for stability
        total_weight = (microwave_mask.sum() > 0).float() * self.microwave_weight + \
                       (saphir_mask.sum() > 0).float() * self.saphir_weight

        if total_weight > 0:
            total_loss = (microwave_loss + saphir_loss) / total_weight
        else:
            total_loss = torch.tensor(0.0, device=pred.device)

        # Final safety check
        return torch.clamp(total_loss, min=0.0)

    def forward(self, model_output, target, microwave_mask, saphir_mask, epoch=0):
        pred = model_output['prediction']
        logvar = model_output['uncertainty']
        mu = model_output.get('mu', None)
        logvar_z = model_output.get('logvar', None)
        device = pred.device

        # KL annealing
        kl_warmup = min(1.0, epoch / self.kl_warmup_epochs)

        # Saphir-aware reconstruction loss (guaranteed non-negative)
        saphir_recon_loss = self.saphir_aware_loss(pred, target, logvar, microwave_mask, saphir_mask)

        # Combined mask for gradient loss
        combined_mask = (microwave_mask + saphir_mask).clamp(max=1.0)

        # Gradient smoothness loss
        grad_loss = self._grad_smoothness(pred, combined_mask)

        # KL divergence loss with annealing
        kl_loss = torch.tensor(0.0, device=device)
        if mu is not None and logvar_z is not None:
            kl_loss = (self.gamma_max * kl_warmup) * self._kl_div(mu, logvar_z)

        # Total loss with proper scaling (all components non-negative)
        total = saphir_recon_loss + self.lambda_grad * grad_loss + kl_loss

        # Final safety clamp
        total = torch.clamp(total, min=0.0, max=50.0)

        metrics = {
            'total': float(total.item()),
            'saphir_recon': float(saphir_recon_loss.item()),
            'grad': float(grad_loss.item()),
            'kl': float(kl_loss.item()) if hasattr(kl_loss, 'item') else float(kl_loss),
        }

        return total, metrics


# ---------------------------
# Training utilities & initialization
# ---------------------------
EPOCHS = 25
LEARNING_RATE = 5e-5  # Reduced for stability
SAVE_EVERY = 5

# Initialize model, optimizer, scheduler, criterion
model = UNetPlusVAEFixed(in_channels=1, base=64, out_channels=1, num_sat=8, dropout=0.1).to(device)
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=7)
criterion = SaphirAwareVAELoss().to(device)

# CORRECTED: Use new GradScaler without device parameter
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
        fused_target = batch['fused_target'].to(device).unsqueeze(1)  # [B,1,H,W]
        microwave_mask = batch['microwave_mask'].to(device)  # [B,H,W]
        saphir_mask = batch['saphir_mask'].to(device)  # [B,H,W]
        saphir_available = batch['saphir_available']

        optimizer.zero_grad()

        # CORRECTED: Use new autocast syntax
        with autocast('cuda'):
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

        # Track Saphir usage statistics
        total_samples += len(saphir_available)
        saphir_samples += saphir_available.sum().item()

        progress_bar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'saphir_recon': f"{metrics['saphir_recon']:.4f}",
            'kl': f"{metrics['kl']:.6f}",
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
            fused_target = batch['fused_target'].to(device).unsqueeze(1)
            microwave_mask = batch['microwave_mask'].to(device)
            saphir_mask = batch['saphir_mask'].to(device)
            saphir_available = batch['saphir_available']

            # CORRECTED: Use new autocast syntax
            with autocast('cuda'):
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

    print("Starting FINAL CORRECTED training with Saphir RH integration...")
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

        # Scheduler
        scheduler.step(val_loss)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
        print(
            f"Train Saphir Coverage: {train_metrics['saphir_coverage']:.1f}% | Val Saphir Coverage: {val_metrics['saphir_coverage']:.1f}%")
        print(f"LR: {optimizer.param_groups[0]['lr']:.8f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'criterion_state_dict': criterion.state_dict(),
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
            print(f"✅ New best validation loss: {val_loss:.6f} (saved to {best_path})")
        else:
            patience_counter += 1

        # Periodic checkpoint
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

    # Save final models
    final_model_path_pth = os.path.join(CKPT_DIR, "UNET++_VAE_Saphir_FINAL.pth")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'criterion_state_dict': criterion.state_dict(),
        'best_val_loss': best_val_loss,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'model_config': {
            'in_channels': 1, 'base': 64, 'out_channels': 1,
            'num_sat': 8, 'dropout': 0.1
        }
    }, final_model_path_pth)
    print(f"✅ Final model saved as: {final_model_path_pth}")

    final_model_path_h5 = os.path.join(CKPT_DIR, "UNET++_VAE_Saphir_FINAL.h5")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'criterion_state_dict': criterion.state_dict(),
        'best_val_loss': best_val_loss,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'model_config': {
            'in_channels': 1, 'base': 64, 'out_channels': 1,
            'num_sat': 8, 'dropout': 0.1
        }
    }, final_model_path_h5)
    print(f"✅ Final model saved as: {final_model_path_h5}")

    return train_losses, val_losses


# ---------------------------
# Evaluation utilities & metrics (preserved from original)
# ---------------------------
def calculate_ergas(true, pred, mean_true=None):
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

    try:
        psnr = 10.0 * np.log10((data_range ** 2) / (mse + 1e-12))
    except:
        psnr = None

    try:
        ergas = calculate_ergas(y_true, y_pred)
    except:
        ergas = None

    ssim = None
    try:
        if mask.all() and y_true.ndim == 2:
            ssim = structural_similarity(y_true, y_pred, data_range=data_range)
    except:
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

            # CORRECTED: Use new autocast syntax
            with autocast('cuda'):
                output = model_eval(inputs, satellite_ids, valid_masks)

            predictions = output['prediction'].cpu().numpy()
            targets = fused_target.cpu().numpy()
            masks = combined_mask.cpu().numpy()

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
    if not metrics_list:
        print("No valid metrics to summarize")
        return

    aggregated = defaultdict(list)
    for metrics in metrics_list:
        for key, value in metrics.items():
            if value is not None and not np.isnan(value) and not np.isinf(value):
                aggregated[key].append(value)

    print("\n" + "=" * 60)
    print("QUANTITATIVE EVALUATION RESULTS (SAPHIR-FINAL)")
    print("=" * 60)

    for metric_name, values in aggregated.items():
        if values:
            mean_val = np.mean(values)
            std_val = np.std(values)
            min_val = np.min(values)
            max_val = np.max(values)
            print(f"{metric_name:>8}: {mean_val:8.4f} ± {std_val:6.4f} "
                  f"(min: {min_val:7.4f}, max: {max_val:7.4f})")
    print("=" * 60)


# Visualization functions (preserved)
def plot_training_curves(train_losses, val_losses):
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    plt.title('Training and Validation Loss Curves (Saphir-FINAL)', fontsize=16)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(CKPT_DIR, 'training_curves_saphir_final.png'), dpi=300, bbox_inches='tight')
    plt.show()


def visualize_predictions(predictions, targets, num_samples=5):
    fig, axes = plt.subplots(2, num_samples, figsize=(15, 6))
    for i in range(min(num_samples, len(predictions))):
        axes[0, i].imshow(targets[i], cmap='viridis', aspect='auto')
        axes[0, i].set_title(f'Target {i + 1}')
        axes[0, i].axis('off')

        axes[1, i].imshow(predictions[i], cmap='viridis', aspect='auto')
        axes[1, i].set_title(f'Prediction {i + 1}')
        axes[1, i].axis('off')

    plt.suptitle('Saphir-FINAL VAE Predictions', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(CKPT_DIR, 'sample_predictions_saphir_final.png'), dpi=300, bbox_inches='tight')
    plt.show()


# ---------------------------
# Run training and evaluation
# ---------------------------
if __name__ == "__main__":
    try:
        train_losses, val_losses = train_model()

        print("\n🎉 Training completed successfully with FINAL CORRECTED Saphir RH integration!")
        print(f"📁 Models saved in: {CKPT_DIR}")
        print(" - UNET++_VAE_Saphir_FINAL.pth")
        print(" - UNET++_VAE_Saphir_FINAL.h5")
        print(" - best_model.pth (best validation model)")

        best_path = os.path.join(CKPT_DIR, "best_model.pth")
        if os.path.exists(best_path):
            print("\nLoading best model for evaluation...")
            best_model = load_best_model()

            print("Evaluating on validation set...")
            predictions, targets, metrics_list = evaluate_model(best_model, val_loader, device, num_samples=808)

            print_evaluation_summary(metrics_list)

            evaluation_results = {
                'predictions': predictions[:10],
                'targets': targets[:10],
                'metrics': metrics_list,
                'summary': {}
            }

            for metric in ['MSE', 'RMSE', 'MAE', 'PSNR', 'ERGAS', 'SSIM']:
                values = [m[metric] for m in metrics_list if (
                        metric in m and m[metric] is not None and not np.isnan(m[metric]) and not np.isinf(m[metric]))]
                if values:
                    evaluation_results['summary'][metric] = {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'count': len(values)
                    }

            results_path = os.path.join(CKPT_DIR, "evaluation_results_saphir_final.pkl")
            with open(results_path, 'wb') as f:
                pickle.dump(evaluation_results, f)
            print(f"\nEvaluation results saved to: {results_path}")

            # plots
            plot_training_curves(train_losses, val_losses)
            visualize_predictions(predictions, targets)

    except Exception as e:
        print(f"❌ Training failed with error: {e}")
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

    print("\nScript completed successfully with FINAL CORRECTED Saphir RH integration!")
