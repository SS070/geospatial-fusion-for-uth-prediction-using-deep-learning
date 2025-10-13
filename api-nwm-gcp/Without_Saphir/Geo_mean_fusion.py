#%%

import numpy as np
import pandas as pd
import xarray as xr
import os
import glob
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from collections import defaultdict

import matplotlib.pyplot as plt
import seaborn as sns

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
import random
import gc
from torch.cuda.amp import autocast, GradScaler

from sklearn.metrics import mean_squared_error, mean_absolute_error
import math
from skimage.metrics import structural_similarity as ssim
import cartopy.crs as ccrs
import random

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True


#%%

# Define base directory and sensor folders
base_dir = r"C:\S\Programming\NRSC\PythonProject\DATA\MICROWAVE_UTH_DATA_NOAA"

# Satellite folder map (you can customize names for clarity)
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

# Master dictionary: {date: {satellite: file_path}}
date_file_index = defaultdict(dict)

for sat, folder in satellite_dirs.items():
    folder_path = os.path.join(base_dir, folder)
    nc_files = glob.glob(os.path.join(folder_path, "*.nc"))

    for file_path in nc_files:
        filename = os.path.basename(file_path)
        try:
            # Extract date from filename (positions 5 to 13: YYYYMMDD)
            date_str = filename[5:13]
            date = datetime.strptime(date_str, "%Y%m%d").date()
            date_file_index[str(date)][sat] = file_path
        except Exception as e:
            print(f" Failed to parse date from {filename}: {e}")

print(f" Indexed {len(date_file_index)} unique dates.")

#%%
sample_day = '2007-03-14'
print(date_file_index[sample_day])

#%%
class UTHMultiSensorVisualaizationDataset(Dataset):
    def __init__(self, date_file_index, transform=None):
        self.date_keys = sorted(date_file_index.keys())
        self.index = date_file_index
        self.transform = transform  # Not used yet, but placeholder for future

    def __len__(self):
        return len(self.date_keys)

    def __getitem__(self, idx):
        date = self.date_keys[idx]
        file_dict = self.index[date]

        inputs = []
        masks = []
        sat_names = []

        for sat, file_path in file_dict.items():
            try:
                ds = xr.open_dataset(file_path)
                uth = ds['uth_mean_ascend_descend'].isel(time=0).values  # (H, W)
                ds.close()

                # Normalize (Z-score, ignoring NaNs)
                mean = np.nanmean(uth)
                std = np.nanstd(uth)
                uth_norm = (uth - mean) / (std + 1e-8)

                # NaN mask: 1 = valid, 0 = missing
                mask = ~np.isnan(uth_norm)

                # Fill NaNs with 0 temporarily (they’ll be masked in loss)
                uth_norm = np.nan_to_num(uth_norm, nan=0.0)

                # Convert to tensor
                uth_tensor = torch.tensor(uth_norm, dtype=torch.float32).unsqueeze(0)  # [1, H, W]
                mask_tensor = torch.tensor(mask.astype(np.float32), dtype=torch.float32).unsqueeze(0)

                inputs.append(uth_tensor)
                masks.append(mask_tensor)
                sat_names.append(sat)

            except Exception as e:
                print(f" Error loading {file_path}: {e}")

        return {
            'date': date,
            'satellites': sat_names,
            'inputs': inputs,         # List of [1, H, W] tensors
            'masks': masks,           # List of [1, H, W] tensors
        }

#%%
dataset =  UTHMultiSensorVisualaizationDataset(date_file_index)
sample = dataset[0]

print(f" Date: {sample['date']}")
print(f"Satellites: {sample['satellites']}")
print(f" #Inputs: {len(sample['inputs'])}")
print(f" Input shape: {sample['inputs'][0].shape}")

#%%

def visualize_sample(sample, idx=0):
    inputs = sample['inputs']
    masks = sample['masks']
    sats = sample['satellites']
    date = sample['date']

    n_inputs = len(inputs)
    fig, axs = plt.subplots(n_inputs, 3, figsize=(12, 4 * n_inputs))
    if n_inputs == 1:
        axs = [axs]  # Ensure it's iterable for one input

    for i in range(n_inputs):
        uth = inputs[i][0].numpy()  # shape [H, W]
        mask = masks[i][0].numpy()  # shape [H, W]

        axs[i][0].imshow(uth, cmap='viridis')
        axs[i][0].set_title(f'{sats[i]}: Normalized UTH\nDate: {date}')
        axs[i][0].axis('off')

        axs[i][1].imshow(mask, cmap='gray')
        axs[i][1].set_title('Mask (1=Valid, 0=NaN)')
        axs[i][1].axis('off')

        axs[i][2].imshow(np.ma.masked_where(mask == 1, uth), cmap='Reds')
        axs[i][2].set_title('Missing Pixels (NaNs)')
        axs[i][2].axis('off')

    plt.tight_layout()
    plt.show()

#%%
# Load a random or specific sample
sample = dataset[0]  # or dataset[random.randint(0, len(dataset))]

# Visualize
visualize_sample(sample)

#%%


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.GELU(),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        scale = self.se(x)
        return x * scale

class ResidualConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_se=True):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels)
        )
        self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1) \
            if in_channels != out_channels else nn.Identity()
        self.use_se = use_se
        self.se = SEBlock(out_channels) if use_se else nn.Identity()

    def forward(self, x):
        res = self.residual_conv(x)
        out = self.conv(x)
        out = self.se(out)
        return F.gelu(out + res)

class SharedEncoder(nn.Module):
    def __init__(self, in_channels=1, base_channels=64):
        super().__init__()
        self.enc1 = ResidualConvBlock(in_channels, base_channels)
        self.enc2 = ResidualConvBlock(base_channels, base_channels * 2)
        self.enc3 = ResidualConvBlock(base_channels * 2, base_channels * 4)
        self.enc4 = ResidualConvBlock(base_channels * 4, base_channels * 8)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool(x1))
        x3 = self.enc3(self.pool(x2))
        x4 = self.enc4(self.pool(x3))
        return [x1, x2, x3, x4]

class AttentionFusion(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(in_channels, 1, kernel_size=1)
        )

    def forward(self, feats):
        B, N, C, H, W = feats.shape
        flat_feats = feats.view(B * N, C, H, W)
        scores = self.attn(flat_feats).view(B, N, 1, H, W)
        attn_weights = F.softmax(scores, dim=1)
        fused = torch.sum(attn_weights * feats, dim=1)
        return fused

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GELU()
        )

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class UNetPlusPlusWithAttention_Optimized(nn.Module):
    def __init__(self, in_channels=1, base_channels=64, out_channels=1):
        super().__init__()
        self.shared_encoder = SharedEncoder(in_channels, base_channels)
        self.fuse1 = AttentionFusion(base_channels)
        self.fuse2 = AttentionFusion(base_channels * 2)
        self.fuse3 = AttentionFusion(base_channels * 4)
        self.fuse4 = AttentionFusion(base_channels * 8)

        self.dec3 = DecoderBlock(base_channels * 8, base_channels * 4, base_channels * 4)
        self.dec2 = DecoderBlock(base_channels * 4, base_channels * 2, base_channels * 2)
        self.dec1 = DecoderBlock(base_channels * 2, base_channels, base_channels)

        self.final = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def forward(self, x):
        B, N, C, H, W = x.shape
        encoded_feats = [self.shared_encoder(x[:, i]) for i in range(N)]

        f1 = torch.stack([feat[0] for feat in encoded_feats], dim=1)
        f2 = torch.stack([feat[1] for feat in encoded_feats], dim=1)
        f3 = torch.stack([feat[2] for feat in encoded_feats], dim=1)
        f4 = torch.stack([feat[3] for feat in encoded_feats], dim=1)

        fused1 = self.fuse1(f1)
        fused2 = self.fuse2(f2)
        fused3 = self.fuse3(f3)
        fused4 = self.fuse4(f4)

        d3 = self.dec3(fused4, fused3)
        d2 = self.dec2(d3, fused2)
        d1 = self.dec1(d2, fused1)

        return self.final(d1)

#%%


all_dates = sorted(date_file_index.keys())  # Already indexed in Step 2
train_dates, val_dates = train_test_split(all_dates, test_size=0.1, random_state=42)

# -------------------------------
# UTH Dataset Loader
# -------------------------------

class UTHDataset(Dataset):
    def __init__(self, dates, date_to_sat_paths, normalize=True):
        self.dates = dates
        self.date_to_sat_paths = date_to_sat_paths
        self.normalize = normalize

    def __len__(self):
        return len(self.dates)

    def __getitem__(self, idx):
        date = self.dates[idx]
        sat_files = self.date_to_sat_paths[date]

        inputs = []
        valid_masks = []

        for sat_name, file_path in sat_files.items():
            try:
                ds = xr.open_dataset(file_path)
                uth = ds['uth_mean_ascend_descend'].values.astype(np.float32)[0]  # shape: [180, 360]
                ds.close()

                # Create valid pixel mask (1 for valid, 0 for NaN)
                valid_mask = ~np.isnan(uth)
                valid_masks.append(torch.tensor(valid_mask, dtype=torch.float32))

                # Replace NaNs with 0 for processing (not interpreted as no data)
                uth = np.nan_to_num(uth, nan=0.0)

                # Normalize [0, 100] to [0, 1] if requested
                if self.normalize:
                    uth = uth / 100.0

                uth_tensor = torch.tensor(uth, dtype=torch.float32).unsqueeze(0)  # [1, 180, 360]
                inputs.append(uth_tensor)

            except Exception as e:
                print(f"Failed to load {file_path}: {e}")

        inputs_tensor = torch.cat(inputs, dim=0)  # [N_sats, 180, 360]
        mask_tensor = torch.stack(valid_masks, dim=0)  # [N_sats, 180, 360]

        return {
            'date': date,
            'inputs': inputs_tensor,     # [N, H, W]
            'valid_masks': mask_tensor,  # [N, H, W]
            'num_inputs': len(inputs)
        }

# -------------------------------
# Create Datasets and DataLoaders
# -------------------------------

train_dataset = UTHDataset(train_dates, date_file_index)
val_dataset   = UTHDataset(val_dates, date_file_index)

train_loader = DataLoader(train_dataset, batch_size=1, pin_memory=True)
val_loader   = DataLoader(val_dataset, batch_size=1, pin_memory=True)

#%%
sample = next(iter(train_loader))
print(" Sanity Check:")
print("Date:", sample['date'])
print("Num Inputs:", sample['num_inputs'])
print("Input shape:", sample['inputs'].shape)
print("Valid mask shape:", sample['valid_masks'].shape)
print("Valid pixel ratio per input:", sample['valid_masks'].mean(dim=(1,2)))

#%%
# Masked Huber Loss (Stricter)
def masked_huber_loss(pred, target, mask, delta=0.3):  # reduced delta
    error = pred - target
    abs_error = torch.abs(error)
    quadratic = torch.minimum(abs_error, torch.tensor(delta).to(pred.device))
    linear = abs_error - quadratic
    loss = 0.5 * quadratic**2 + delta * linear
    masked_loss = (loss * mask).sum() / mask.sum()
    return masked_loss

# Masked Charbonnier Loss
def masked_charbonnier_loss(pred, target, mask, epsilon=1e-3):
    diff = pred - target
    loss = torch.sqrt(diff**2 + epsilon**2)
    masked_loss = (loss * mask).sum() / mask.sum()
    return masked_loss

# Edge-aware TV Loss
def edge_aware_tv_loss(pred, mask):
    dx = pred[:, :, :, :-1] - pred[:, :, :, 1:]
    dy = pred[:, :, :-1, :] - pred[:, :, 1:, :]

    edge_x = torch.abs(mask[:, :, :, :-1] - mask[:, :, :, 1:])
    edge_y = torch.abs(mask[:, :, :-1, :] - mask[:, :, 1:, :])

    loss = (torch.abs(dx) * edge_x).sum() + (torch.abs(dy) * edge_y).sum()
    norm = (edge_x.sum() + edge_y.sum()).clamp(min=1.0)
    return loss / norm

# Combined Loss
def combined_geospatial_loss(pred, target, mask, epoch=None, decay_start_epoch=10):
    """
    pred, target, mask: [B, 1, H, W]
    epoch: current epoch number (int)
    decay_start_epoch: epoch after which beta is reduced
    """

    alpha = 0.6   # Huber
    beta = 0.25   # TV (stronger than before)
    gamma = 0.15  # Charbonnier

    # Adaptive TV decay
    if epoch is not None:
        if epoch >= decay_start_epoch:
            beta *= 0.6
        if epoch >= decay_start_epoch + 10:
            beta *= 0.3

    loss_huber = masked_huber_loss(pred, target, mask)
    loss_charb = masked_charbonnier_loss(pred, target, mask)
    loss_tv = edge_aware_tv_loss(pred, mask)

    total_loss = alpha * loss_huber + beta * loss_tv + gamma * loss_charb
    return total_loss
#%%
def train_model(model, train_loader, val_loader, optimizer, scheduler, num_epochs=25, device='cuda'):
    scaler = GradScaler()

    for epoch in range(num_epochs):
        model.train()
        train_loss_total = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Training]", leave=False):
            inputs = batch['inputs'].to(device, non_blocking=True)
            valid_masks = batch['valid_masks'].to(device, non_blocking=True)

            inputs = inputs.unsqueeze(2)
            valid_masks = valid_masks.unsqueeze(2)

            valid_mask_sum = valid_masks.sum(dim=1).clamp(min=1e-6)
            target = (inputs * valid_masks).sum(dim=1) / valid_mask_sum
            combined_mask = (valid_masks.sum(dim=1) > 0).float()

            optimizer.zero_grad()

            with autocast():
                output = model(inputs)
                loss = combined_geospatial_loss(output, target, combined_mask, epoch)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss_total += loss.item()

            del inputs, valid_masks, target, combined_mask, output, loss
            torch.cuda.empty_cache()

        avg_train_loss = train_loss_total / len(train_loader)

        # -------------------
        # Validation
        # -------------------
        model.eval()
        val_loss_total = 0.0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Validation]", leave=False):
                inputs = batch['inputs'].to(device, non_blocking=True)
                valid_masks = batch['valid_masks'].to(device, non_blocking=True)

                inputs = inputs.unsqueeze(2)
                valid_masks = valid_masks.unsqueeze(2)

                valid_mask_sum = valid_masks.sum(dim=1).clamp(min=1e-6)
                target = (inputs * valid_masks).sum(dim=1) / valid_mask_sum
                combined_mask = (valid_masks.sum(dim=1) > 0).float()

                with autocast():
                    output = model(inputs)
                    loss = combined_geospatial_loss(output, target, combined_mask, epoch)

                val_loss_total += loss.item()

                del inputs, valid_masks, target, combined_mask, output, loss
                torch.cuda.empty_cache()

        avg_val_loss = val_loss_total / len(val_loader)
        scheduler.step(avg_val_loss)

        print(f"[Epoch {epoch+1:02d}/{num_epochs}] Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")

        # Final cleanup
        del batch
        gc.collect()
        torch.cuda.empty_cache()

#%%
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = UNetPlusPlusWithAttention_Optimized(in_channels=1, out_channels=1).to(device)

# Optimizer: AdamW with weight decay
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

# Scheduler: Cosine Annealing with Warm Restarts (stable long training)
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)

# Use this during training loop (Step 3)
# In your training loop, replace scheduler.step(val_loss) with:
# scheduler.step(epoch + batch_idx / total_batches)


train_model(model, train_loader, val_loader, optimizer, scheduler, num_epochs=36, device=device)

#%%
def compute_metrics(pred, target, mask):
    pred = pred.detach().cpu().numpy()
    target = target.detach().cpu().numpy()
    mask = mask.detach().cpu().numpy()

    pred = pred * mask
    target = target * mask

    valid_pixels = mask > 0

    pred_valid = pred[valid_pixels]
    target_valid = target[valid_pixels]

    mse_val = mean_squared_error(target_valid, pred_valid)
    mae_val = mean_absolute_error(target_valid, pred_valid)
    rmse_val = math.sqrt(mse_val)

    # PSNR
    max_pixel = 1.0  # since normalized to [0,1]
    psnr_val = 20 * math.log10(max_pixel / math.sqrt(mse_val + 1e-8))

    # SSIM (we take the center crop to avoid border artifacts)
    pred_img = pred[0, :, :]
    target_img = target[0, :, :]
    mask_img = mask[0, :, :]

    # Masked SSIM
    try:
        ssim_val = ssim(target_img, pred_img, data_range=1.0, multichannel=False)
    except:
        ssim_val = float('nan')

    return {
        'RMSE': rmse_val,
        'MAE': mae_val,
        'MSE': mse_val,
        'PSNR': psnr_val,
        'SSIM': ssim_val
    }

#%%
def evaluate_model(model, val_loader, device='cuda'):
    model.eval()
    all_metrics = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            inputs = batch['inputs'].to(device)           # [B, N, H, W]
            valid_masks = batch['valid_masks'].to(device) # [B, N, H, W]
            B, N, H, W = inputs.shape

            inputs = inputs.unsqueeze(2)           # [B, N, 1, H, W]
            valid_masks = valid_masks.unsqueeze(2) # [B, N, 1, H, W]

            valid_mask_sum = valid_masks.sum(dim=1)
            valid_mask_sum = torch.clamp(valid_mask_sum, min=1e-6)
            target = (inputs * valid_masks).sum(dim=1) / valid_mask_sum  # [B, 1, H, W]

            output = model(inputs)
            combined_mask = (valid_masks.sum(dim=1) > 0).float()

            for i in range(B):  # batch size usually 1
                metrics = compute_metrics(output[i], target[i], combined_mask[i])
                all_metrics.append(metrics)

            del batch, inputs, valid_masks, valid_mask_sum, target, output, combined_mask
            gc.collect()
            torch.cuda.empty_cache()

    # Average over all samples
    avg_metrics = {
        key: sum(d[key] for d in all_metrics) / len(all_metrics)
        for key in all_metrics[0]
    }

    print("\n Quantitative Evaluation on Validation Set:")
    for k, v in avg_metrics.items():
        print(f"{k}: {v:.4f}")

    return avg_metrics

#%%
evaluate_model(model, val_loader, device=device)

#%%
import matplotlib.pyplot as plt

def visualize_prediction(model, data_loader, device='cuda', num_samples=3):
    model.eval()
    samples_shown = 0

    with torch.no_grad():
        for batch in data_loader:
            inputs = batch['inputs'].to(device)         # [B, N, H, W]
            valid_masks = batch['valid_masks'].to(device)

            B, N, H, W = inputs.shape
            inputs = inputs.unsqueeze(2)                # -> [B, N, 1, H, W]
            valid_masks = valid_masks.unsqueeze(2)

            # Compute target (masked mean)
            valid_mask_sum = valid_masks.sum(dim=1)
            valid_mask_sum = torch.clamp(valid_mask_sum, min=1e-6)
            target = (inputs * valid_masks).sum(dim=1) / valid_mask_sum  # [B, 1, H, W]

            # Prediction
            output = model(inputs)  # [B, 1, H, W]

            # Plot a few examples
            for i in range(min(B, num_samples - samples_shown)):
                fig, axs = plt.subplots(1, 4, figsize=(16, 4))

                input_mean = (inputs[i] * valid_masks[i]).sum(dim=0) / torch.clamp(valid_masks[i].sum(dim=0), min=1e-6)

                axs[0].imshow(input_mean.squeeze().cpu(), cmap='viridis')
                axs[0].set_title("Input Mean (Masked)")
                axs[0].axis('off')

                axs[1].imshow(target[i].squeeze().cpu(), cmap='viridis')
                axs[1].set_title("Target")
                axs[1].axis('off')

                axs[2].imshow(output[i].squeeze().cpu(), cmap='viridis')
                axs[2].set_title("Prediction")
                axs[2].axis('off')

                axs[3].imshow((target[i] - output[i]).squeeze().abs().cpu(), cmap='hot')
                axs[3].set_title("Error Map (Abs Diff)")
                axs[3].axis('off')

                plt.tight_layout()
                plt.show()

                samples_shown += 1
                if samples_shown >= num_samples:
                    return

#%%
visualize_prediction(model, val_loader, device=device, num_samples=3)

#%%
# Save the entire model
torch.save(model, "unetplusplus_Optim_model_full.pth")
torch.save(model, "unetplusplus_Optim_model_full.h5")

#%%
# Save only model state_dict (weights)
torch.save(model.state_dict(), "unetplusplus_Optim_model_weights.pth")

#%%
# Denorm
#%%
def compute_metrics(pred, target, mask, norm_factor=100.0):
    # Denormalize
    pred = pred * norm_factor
    target = target * norm_factor

    pred = pred.detach().cpu().numpy()
    target = target.detach().cpu().numpy()
    mask = mask.detach().cpu().numpy()

    pred = pred * mask
    target = target * mask

    valid_pixels = mask > 0

    pred_valid = pred[valid_pixels]
    target_valid = target[valid_pixels]

    mse_val = mean_squared_error(target_valid, pred_valid)
    mae_val = mean_absolute_error(target_valid, pred_valid)
    rmse_val = math.sqrt(mse_val)

    # PSNR
    max_pixel = norm_factor  # since values are in [0, norm_factor]
    psnr_val = 20 * math.log10(max_pixel / math.sqrt(mse_val + 1e-8))

    # SSIM (still between 0-1 range)
    pred_img = pred[0, :, :]
    target_img = target[0, :, :]
    mask_img = mask[0, :, :]

    try:
        ssim_val = ssim(target_img, pred_img, data_range=norm_factor, multichannel=False)
    except:
        ssim_val = float('nan')

    return {
        'RMSE': rmse_val,
        'MAE': mae_val,
        'MSE': mse_val,
        'PSNR': psnr_val,
        'SSIM': ssim_val
    }

#%%
def evaluate_model(model, val_loader, device='cuda', norm_factor=100.0):
    model.eval()
    all_metrics = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            inputs = batch['inputs'].to(device)           # [B, N, H, W]
            valid_masks = batch['valid_masks'].to(device) # [B, N, H, W]
            B, N, H, W = inputs.shape

            inputs = inputs.unsqueeze(2)           # [B, N, 1, H, W]
            valid_masks = valid_masks.unsqueeze(2) # [B, N, 1, H, W]

            valid_mask_sum = valid_masks.sum(dim=1)
            valid_mask_sum = torch.clamp(valid_mask_sum, min=1e-6)
            target = (inputs * valid_masks).sum(dim=1) / valid_mask_sum  # [B, 1, H, W]

            output = model(inputs)
            combined_mask = (valid_masks.sum(dim=1) > 0).float()

            for i in range(B):
                metrics = compute_metrics(output[i], target[i], combined_mask[i], norm_factor=norm_factor)
                all_metrics.append(metrics)

            del batch, inputs, valid_masks, valid_mask_sum, target, output, combined_mask
            gc.collect()
            torch.cuda.empty_cache()
    # Average over all samples
    avg_metrics = {
        key: sum(d[key] for d in all_metrics) / len(all_metrics)
        for key in all_metrics[0]
    }

    print("\n Quantitative Evaluation on Validation Set (Denormalized):")
    for k, v in avg_metrics.items():
        print(f"{k}: {v:.4f}")

        gc.collect()
        torch.cuda.empty_cache()

    return avg_metrics

#%%
evaluate_model(model, val_loader, device=device, norm_factor=100.0)

#%%
