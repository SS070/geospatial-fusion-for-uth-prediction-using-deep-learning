import os
import numpy as np
import xarray as xr
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from skimage.metrics import structural_similarity as ssim_metric
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from tqdm.notebook import tqdm
import random

#%%
# Base directory
base_dir = r"/MICROWAVE_UTH_DATA_NOAA"
sensor_dirs = {
    "Metop_A": os.path.join(base_dir, "MHS_Metop_A_2007_2021"),
    "Metop_B": os.path.join(base_dir, "MHS_Metop_B_2013_2021"),
    "NOAA_18": os.path.join(base_dir, "MHS_NOAA_18_2006_2018"),
    "NOAA_19": os.path.join(base_dir, "MHS_NOAA_19_2016_2021"),
}
sensor_files = {
    k: sorted([os.path.join(v, f) for f in os.listdir(v) if f.endswith(".nc")])
    for k,v in sensor_dirs.items()
}

# Load one variable from all files
def load_sensor_stack(file_list, variable="uth_mean_ascend_descend"):
    uth_arrays = []
    for f in file_list:
        ds = xr.open_dataset(f)
        uth = ds[variable]
        if "time" in uth.dims and uth.sizes["time"] == 1:
            uth = uth.squeeze("time")
        uth_arrays.append(uth)
        ds.close()
    return xr.concat(uth_arrays, dim="time")

sensor_data = {s: load_sensor_stack(files) for s, files in sensor_files.items()}
ref_lat = sensor_data["Metop_A"]["lat"]
ref_lon = sensor_data["Metop_A"]["lon"]
sensor_data_aligned = {s: da.interp(lat=ref_lat, lon=ref_lon, method="nearest") for s, da in sensor_data.items()}
arrays = [sensor_data_aligned[s].values for s in sensor_dirs.keys()]
data_4d = np.stack(arrays, axis=-1)  # (time, lat, lon, sensor)

# Normalize
means = []
stds = []
for i in range(data_4d.shape[-1]):
    vals = data_4d[...,i]
    means.append(np.nanmean(vals))
    stds.append(np.nanstd(vals))

data_4d_norm = (data_4d - means) / stds

#%%
PATCH_SIZE = (64,64)
MASK_RATIO = 0.3

def pad_patch(patch, target_shape):
    H,W,C = patch.shape
    out = np.zeros((target_shape[0], target_shape[1], C), dtype=patch.dtype)
    out[:H,:W,:] = patch
    return out

def extract_patches(data, patch_size):
    time_dim, lat_dim, lon_dim, num_sensors = data.shape
    patches = []
    for t in range(time_dim):
        for i in range(0, lat_dim, patch_size[0]):
            for j in range(0, lon_dim, patch_size[1]):
                p = data[t, i:i+patch_size[0], j:j+patch_size[1], :]
                p = pad_patch(p, patch_size)
                patches.append(p)
    return patches

patches = extract_patches(data_4d_norm, PATCH_SIZE)
train_p, val_p = train_test_split(patches, test_size=0.1, random_state=42)

#%%
class PatchDataset(Dataset):
    def __init__(self, patches, mask_ratio=0.3):
        self.patches = patches
        self.mask_ratio = mask_ratio

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        p = torch.tensor(self.patches[idx], dtype=torch.float32)
        valid_mask = torch.isfinite(p).all(dim=-1).float()
        p[torch.isnan(p)] = 0.0

        train_mask = torch.ones((p.shape[0], p.shape[1]), dtype=torch.float32)
        n_pixels = train_mask.numel()
        n_mask = int(self.mask_ratio * n_pixels)
        mask_indices = random.sample(range(n_pixels), n_mask)
        train_mask.view(-1)[mask_indices] = 0

        x = p.clone()
        x[train_mask==0] = 0.0

        return (
            x.permute(2,0,1),
            p.permute(2,0,1),
            train_mask,
            valid_mask
        )

train_ds = PatchDataset(train_p, mask_ratio=MASK_RATIO)
val_ds = PatchDataset(val_p, mask_ratio=MASK_RATIO)

train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=4)

#%%
import torch
import torch.nn as nn

def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )

class UNetPP(nn.Module):
    def __init__(self, in_channels=4, out_channels=4, filters=32):
        super(UNetPP, self).__init__()

        self.conv0_0 = conv_block(in_channels, filters)
        self.conv1_0 = conv_block(filters, filters * 2)
        self.conv2_0 = conv_block(filters * 2, filters * 4)
        self.conv3_0 = conv_block(filters * 4, filters * 8)
        self.conv4_0 = conv_block(filters * 8, filters * 16)

        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        # Level 1
        self.conv0_1 = conv_block(filters + filters * 2, filters)
        self.conv1_1 = conv_block(filters * 2 + filters * 4, filters * 2)
        self.conv2_1 = conv_block(filters * 4 + filters * 8, filters * 4)
        self.conv3_1 = conv_block(filters * 8 + filters * 16, filters * 8)

        # Level 2
        self.conv0_2 = conv_block(filters * 2 + filters, filters)
        self.conv1_2 = conv_block(filters * 4 + filters * 2, filters * 2)
        self.conv2_2 = conv_block(filters * 8 + filters * 4, filters * 4)

        # Level 3
        self.conv0_3 = conv_block(filters * 3 + filters, filters)
        self.conv1_3 = conv_block(filters * 6 + filters * 2, filters * 2)

        # Level 4
        self.conv0_4 = conv_block(filters * 4, filters)  # **Fixed input channels**

        self.final = nn.Conv2d(filters, out_channels, kernel_size=1)

    def forward(self, x):
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], dim=1))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], dim=1))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], dim=1))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], dim=1))

        x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], dim=1))
        x1_2 = self.conv1_2(torch.cat([x1_0, self.up(x2_1)], dim=1))
        x0_2 = self.conv0_2(torch.cat([x0_0, self.up(x1_1)], dim=1))

        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, self.up(x2_2)], dim=1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, self.up(x1_2)], dim=1))

        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3], dim=1))

        out = self.final(x0_4)
        return out

#%%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNetPP(in_channels=4, out_channels=4).to(device)
#%%
from pytorch_msssim import ssim

def masked_mae(pred, target, mask, valid_mask):
    loss_mask = (mask == 0) & (valid_mask == 1)
    loss_mask = loss_mask.unsqueeze(1).expand_as(pred)
    abs_err = torch.abs(pred - target) * loss_mask
    return abs_err.sum() / loss_mask.sum().clamp(min=1)

def masked_mse(pred, target, mask, valid_mask):
    loss_mask = (mask == 0) & (valid_mask == 1)
    loss_mask = loss_mask.unsqueeze(1).expand_as(pred)
    sq_err = (pred - target) ** 2 * loss_mask
    return sq_err.sum() / loss_mask.sum().clamp(min=1)

def masked_rmse(pred, target, mask, valid_mask):
    return torch.sqrt(masked_mse(pred, target, mask, valid_mask))

from pytorch_msssim import ssim

def masked_mae_ssim(pred, target, mask, valid_mask, alpha=0.8):
    mae = masked_mae(pred, target, mask, valid_mask)
    ssim_loss = 1 - ssim(pred, target, data_range=1.0, size_average=True)
    return alpha * mae + (1 - alpha) * ssim_loss


optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

#%%
import time

NUM_EPOCHS = 750

train_losses = []
val_losses = []

# train_maes = []
# train_rmses = []
train_mses = []

val_maes = []
val_rmses = []
val_mses = []

for epoch in range(NUM_EPOCHS):
    start_time = time.time()

    # ---------------- TRAINING ----------------
    model.train()
    train_loss_epoch = 0.0
    # train_mae_epoch = 0.0
    # train_rmse_epoch = 0.0
    train_mse_epoch = 0.0
    n_train_batches = 0

    for xb, yb, maskb, valid_maskb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        maskb, valid_maskb = maskb.to(device), valid_maskb.to(device)

        optimizer.zero_grad()
        preds = model(xb)
        loss = masked_mae_ssim(preds, yb, maskb, valid_maskb)
        loss.backward()
        optimizer.step()

        train_loss_epoch += loss.item()
        # train_mae_epoch += masked_mae(preds, yb, maskb, valid_maskb).item()
        # train_rmse_epoch += masked_rmse(preds, yb, maskb, valid_maskb).item()
        train_mse_epoch += masked_mse(preds, yb, maskb, valid_maskb).item()
        n_train_batches += 1

    mean_train_loss = train_loss_epoch / n_train_batches
    # mean_train_mae = train_mae_epoch / n_train_batches
    # mean_train_rmse = train_rmse_epoch / n_train_batches
    mean_train_mse = train_mse_epoch / n_train_batches

    train_losses.append(mean_train_loss)
    # train_maes.append(mean_train_mae)
    # train_rmses.append(mean_train_rmse)
    train_mses.append(mean_train_mse)

    # ---------------- VALIDATION ----------------
    model.eval()
    val_loss_epoch = 0.0
    val_mae_epoch = 0.0
    val_rmse_epoch = 0.0
    val_mse_epoch = 0.0
    n_val_batches = 0

    with torch.no_grad():
        for xb, yb, maskb, valid_maskb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            maskb, valid_maskb = maskb.to(device), valid_maskb.to(device)

            preds = model(xb)
            loss = masked_mae_ssim(preds, yb, maskb, valid_maskb)

            val_loss_epoch += loss.item()
            val_mae_epoch += masked_mae(preds, yb, maskb, valid_maskb).item()
            val_rmse_epoch += masked_rmse(preds, yb, maskb, valid_maskb).item()
            val_mse_epoch += masked_mse(preds, yb, maskb, valid_maskb).item()
            n_val_batches += 1

    mean_val_loss = val_loss_epoch / n_val_batches
    mean_val_mae = val_mae_epoch / n_val_batches
    mean_val_rmse = val_rmse_epoch / n_val_batches
    mean_val_mse = val_mse_epoch / n_val_batches

    val_losses.append(mean_val_loss)
    val_maes.append(mean_val_mae)
    val_rmses.append(mean_val_rmse)
    val_mses.append(mean_val_mse)

    elapsed = time.time() - start_time
    print(f"Epoch {epoch+1:03d}/{NUM_EPOCHS} — "
          f"Train Loss: {mean_train_loss:.5f} | Val Loss: {mean_val_loss:.5f} | "
          f"Val MAE: {mean_val_mae:.4f} | "
          f"Val RMSE: {mean_val_rmse:.4f} | ")
          # f"Time: {elapsed:.1f}s")

#%%
torch.save(model, "UNet++_GEO_AUG2018.pth")
torch.save(model, "UNet++_GEO_AUG2018.h5")
#%%
model.eval()

n_samples = 3  # How many patches to visualize

for idx in range(n_samples):
    # Get a sample
    inputs, targets, train_mask, valid_mask = val_ds[idx]

    # Add batch dimension and send to GPU
    inputs = inputs.unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(inputs).cpu().squeeze(0)

    # Convert to numpy
    inputs_np = inputs.cpu().squeeze(0).numpy()
    targets_np = targets.numpy()
    pred_np = pred.numpy()
    train_mask_np = train_mask.numpy()
    valid_mask_np = valid_mask.numpy()

    # For each sensor channel
    channels = ["Metop_A", "Metop_B", "NOAA_18", "NOAA_19"]

    for ch in range(4):
        fig, axs = plt.subplots(1, 3, figsize=(15, 4))

        axs[0].imshow(targets_np[ch], cmap="viridis", vmin=-2, vmax=2)
        axs[0].set_title(f"Ground Truth ({channels[ch]})")
        axs[0].axis("off")

        axs[1].imshow(pred_np[ch], cmap="viridis", vmin=-2, vmax=2)
        axs[1].set_title("Prediction")
        axs[1].axis("off")

        axs[2].imshow(train_mask_np, cmap="gray")
        axs[2].set_title("Masked Pixels (black)")
        axs[2].axis("off")

        plt.suptitle(f"Patch {idx} - {channels[ch]}")
        plt.tight_layout()
        plt.show()
#%%
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

def compute_patch_metrics(pred, target, mask, sensor_idx):
    """
    pred, target: (H,W,C)
    mask: (H,W)
    sensor_idx: 0 to 3
    """
    pred_ch = pred[:,:,sensor_idx]
    target_ch = target[:,:,sensor_idx]

    # Evaluate only on masked pixels
    eval_mask = (mask==0) & np.isfinite(target_ch)

    pred_valid = pred_ch[eval_mask]
    target_valid = target_ch[eval_mask]

    if pred_valid.size == 0:
        return np.nan, np.nan, np.nan

    mse = mean_squared_error(target_valid, pred_valid)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(target_valid, pred_valid)

    return mse, rmse, mae

#%%
# Pick one patch
inputs, targets, train_mask, valid_mask = val_ds[0]

inputs = inputs.unsqueeze(0).to(device)
with torch.no_grad():
    pred = model(inputs).cpu().squeeze(0)

pred_np = pred.numpy().transpose(1,2,0)
target_np = targets.numpy().transpose(1,2,0)
mask_np = train_mask.numpy()

# For each sensor
for i, sensor in enumerate(["Metop_A", "Metop_B", "NOAA_18", "NOAA_19"]):
    mse, rmse, mae = compute_patch_metrics(pred_np, target_np, mask_np, i)
    print(f"{sensor}: MSE={mse:.4f} | RMSE={rmse:.4f} | MAE={mae:.4f}")

#%%
import cartopy.crs as ccrs
import cartopy.feature as cfeature

def plot_patch_robinson(pred, target, mask, sensor_names, patch_idx=0):
    """
    pred, target: tensors (B, C, H, W)
    mask: tensors (B, H, W)
    """
    p = pred[patch_idx].detach().cpu().numpy()
    t = target[patch_idx].detach().cpu().numpy()
    m = mask[patch_idx].detach().cpu().numpy()

    num_sensors = len(sensor_names)

    fig, axs = plt.subplots(2, num_sensors, figsize=(4*num_sensors, 8),
                            subplot_kw={"projection": ccrs.Robinson()})

    # Flatten axs in case num_sensors == 1
    axs = np.atleast_2d(axs)

    # Create a common normalization (vmin/vmax) for consistent color mapping
    vmin = min(p.min(), t.min())
    vmax = max(p.max(), t.max())

    for i, sensor in enumerate(sensor_names):
        # Target (Top Row)
        ax = axs[0, i]
        ax.set_title(f"Target - {sensor}")
        im = ax.imshow(
            t[i],
            transform=ccrs.PlateCarree(),
            extent=[-180,180,-90,90],
            cmap="viridis",
            vmin=vmin,
            vmax=vmax
        )
        ax.coastlines()
        ax.add_feature(cfeature.BORDERS, linestyle=':')

        # Prediction (Bottom Row)
        ax = axs[1, i]
        ax.set_title(f"Prediction - {sensor}")
        im = ax.imshow(
            p[i],
            transform=ccrs.PlateCarree(),
            extent=[-180,180,-90,90],
            cmap="viridis",
            vmin=vmin,
            vmax=vmax
        )
        ax.coastlines()
        ax.add_feature(cfeature.BORDERS, linestyle=':')

    # Add a single colorbar below all plots
    cbar_ax = fig.add_axes([0.15, 0.05, 0.7, 0.02])  # wider and more centered
    fig.colorbar(im, cax=cbar_ax, orientation="horizontal")

    plt.tight_layout()  # leave space for the colorbar
    plt.show()

#%%
# Take a batch
batch = next(iter(val_loader))
inputs, targets, train_mask, valid_mask = [b.to(device) for b in batch]
outputs = model(inputs)

sensor_names = ["Metop_A", "Metop_B", "NOAA_18", "NOAA_19"]

plot_patch_robinson(outputs, targets, train_mask, sensor_names, patch_idx=0)

#%%
from skimage.metrics import structural_similarity as ssim_metric
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from sklearn.metrics import mean_squared_error, mean_absolute_error

import numpy as np

# Containers for all patches
ssim_list = []
psnr_list = []
mse_list = []
rmse_list = []
mae_list = []

model.eval()

for idx in range(len(val_ds)):
    inputs, targets, train_mask, valid_mask = val_ds[idx]

    with torch.no_grad():
        pred = model(inputs.unsqueeze(0).to(device)).cpu().squeeze(0)  # (C,H,W)

    mask = (train_mask.numpy() == 0) & (valid_mask.numpy() == 1)  # which pixels were masked

    pred_np = pred.numpy()
    targets_np = targets.numpy()

    # Loop over sensors
    for ch in range(pred_np.shape[0]):
        gt = targets_np[ch]
        pr = pred_np[ch]

        # Only evaluate where mask==1
        mask_ch = mask
        if mask_ch.sum() == 0:
            continue

        gt_valid = gt[mask_ch]
        pr_valid = pr[mask_ch]

        # Compute metrics
        mse = np.mean((gt_valid - pr_valid) **2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(gt_valid - pr_valid))

        # SSIM and PSNR require 2D arrays (full patch)
        # We'll set non-masked regions to ground truth so they don't penalize
        gt_full = gt.copy()
        pr_full = pr.copy()
        gt_full[~mask_ch] = gt[~mask_ch]
        pr_full[~mask_ch] = gt[~mask_ch]  # mask with ground truth for fairness

        ssim_val = ssim_metric(gt_full, pr_full, data_range=gt_full.max() - gt_full.min())
        psnr_val = psnr_metric(gt_full, pr_full, data_range=gt_full.max() - gt_full.min())

        mse_list.append(mse)
        rmse_list.append(rmse)
        mae_list.append(mae)
        ssim_list.append(ssim_val)
        psnr_list.append(psnr_val)

# =============================
# FINAL REPORT
# =============================
print("\n=== Validation Metrics (Masked Regions) ===")
print(f"Mean MSE:   {np.mean(mse_list):.4f}")
print(f"Mean RMSE:  {np.mean(rmse_list):.4f}")
print(f"Mean MAE:   {np.mean(mae_list):.4f}")
print(f"Mean SSIM:  {np.mean(ssim_list):.4f}")
print(f"Mean PSNR:  {np.mean(psnr_list):.2f} dB")

#%%
means = []
stds = []

for i in range(data_4d.shape[-1]):
    sensor_data = data_4d[..., i]
    mean_val = np.nanmean(sensor_data)
    std_val = np.nanstd(sensor_data)
    means.append(mean_val)
    stds.append(std_val)

# Convert to numpy arrays for convenience
means = np.array(means)
stds = np.array(stds)

print("Means per sensor:", means)
print("Stds per sensor:", stds)

#%%
# Your known normalized metrics
mean_rmse_norm = 0.1418
mean_mae_norm = 0.0979

# Your per-sensor stds
stds = [18.108423, 18.000591, 18.094555, 15.177217]

sensors = ["Metop_A", "Metop_B", "NOAA_18", "NOAA_19"]

print("==== Error Metrics per Sensor (in % UTH) ====")
for sensor, std in zip(sensors, stds):
    rmse_real = mean_rmse_norm * std
    mae_real = mean_mae_norm * std
    print(f"{sensor}: RMSE = {rmse_real:.2f}% UTH | MAE = {mae_real:.2f}% UTH")

#%%
torch.save(model.state_dict(), 'Unet++_GEO_AUG2018_weights.pth')
#%%
