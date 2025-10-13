import os
import glob
import xarray as xr
import numpy as np
import pandas as pd
import torch
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

#%%
# ===========================================================
# STEP 2: Define dataset directories
# ===========================================================

base_dir = r"/MICROWAVE_UTH_DATA_NOAA"

sensor_dirs = {
    "Metop_A": os.path.join(base_dir, "MHS_Metop_A_2007_2021"),
    "Metop_B": os.path.join(base_dir, "MHS_Metop_B_2013_2021"),
    "NOAA_18": os.path.join(base_dir, "MHS_NOAA_18_2006_2018"),
    "NOAA_19": os.path.join(base_dir, "MHS_NOAA_19_2016_2021"),
}

# Gather .nc files for each sensor
sensor_files = {
    sensor: sorted(glob.glob(os.path.join(path, "*.nc")))
    for sensor, path in sensor_dirs.items()
}

# Print count to confirm
for sensor, files in sensor_files.items():
    print(f"{sensor}: {len(files)} files")

#%%
# ===========================================================
# STEP 3: Load all NetCDF files into xarray DataArrays
# ===========================================================

def load_sensor_data(files, variable="uth_mean_ascend_descend"):
    uth_list = []
    time_list = []
    for file in tqdm(files):
        ds = xr.open_dataset(file)
        uth = ds[variable]

        # Some files have singleton time dimension
        if "time" in uth.dims and uth.sizes["time"] == 1:
            uth = uth.squeeze("time")

        # Append array and timestamp
        uth_list.append(uth)
        tval = ds["time"].values
        if isinstance(tval, np.ndarray) and tval.size == 1:
            tval = tval.item()
        time_list.append(pd.to_datetime(tval))
        ds.close()

    # Concatenate along time
    uth_stack = xr.concat(uth_list, dim="time")
    uth_stack["time"] = time_list
    return uth_stack

# Load data per sensor
sensor_data = {
    sensor: load_sensor_data(files)
    for sensor, files in sensor_files.items()
}

#%%
# ===========================================================
# STEP 4: Validate lat/lon grids across sensors
# ===========================================================

# Pick reference grid
ref_lat = sensor_data["Metop_A"]["lat"]
ref_lon = sensor_data["Metop_A"]["lon"]

for sensor, da in sensor_data.items():
    if not np.allclose(da["lat"], ref_lat):
        print(f"WARNING: Latitude mismatch in {sensor}")
    if not np.allclose(da["lon"], ref_lon):
        print(f"WARNING: Longitude mismatch in {sensor}")

#%%
# ===========================================================
# STEP 5: Regrid to reference lat/lon
# ===========================================================

sensor_data_aligned = {
    sensor: da.interp(lat=ref_lat, lon=ref_lon, method="nearest")
    for sensor, da in sensor_data.items()
}

#%%
# ===========================================================
# STEP 6: Convert to 4D numpy array
# Shape: (time, lat, lon, sensor)
# ===========================================================

arrays = [sensor_data_aligned[sensor].values for sensor in sensor_dirs.keys()]
data_4d = np.stack(arrays, axis=-1)  # (time, lat, lon, sensor)

print("Data shape:", data_4d.shape)

#%%
# ===========================================================
# STEP 7: Quick visualization of one day, one sensor
# ===========================================================

day_idx = 0
sensor_idx = 0

plt.figure(figsize=(12,6))
plt.imshow(data_4d[day_idx,:,:,sensor_idx], cmap="viridis")
plt.colorbar()
plt.title("UTH  - Day 0 - Metop_A")
plt.show()

#%%
import numpy as np
import matplotlib.pyplot as plt

sensor_names = ["Metop_A", "Metop_B", "NOAA_18", "NOAA_19"]

missing_maps = []

for i, name in enumerate(sensor_names):
    # Mask where data is NaN
    mask_nan = np.isnan(data_4d[:,:,:,i])

    # Compute fraction of times NaN at each pixel
    missing_fraction = mask_nan.mean(axis=0)
    missing_maps.append(missing_fraction)

    # Plot
    plt.figure(figsize=(10,5))
    plt.imshow(missing_fraction, cmap="viridis")
    plt.colorbar(label="Fraction Missing")
    plt.title(f"Missing Data Fraction - {name}")
    plt.xlabel("Longitude pixels")
    plt.ylabel("Latitude pixels")
    plt.tight_layout()
    plt.show()

#%%
plt.figure(figsize=(12,6))

for i, name in enumerate(sensor_names):
    data_flat = data_4d[:,:,:,i].flatten()
    data_flat = data_flat[~np.isnan(data_flat)]  # drop NaNs

    plt.hist(data_flat, bins=50, alpha=0.5, label=name)

plt.xlabel("UTH Value")
plt.ylabel("Frequency")
plt.title("Histogram of UTH Values per Sensor")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#%%
plt.figure(figsize=(12,6))

for i, name in enumerate(sensor_names):
    # Compute daily mean (ignoring NaNs)
    daily_mean = np.nanmean(data_4d[:,:,:,i], axis=(1,2))
    plt.plot(daily_mean, label=name)

plt.xlabel("Day Index (0=Aug 1)")
plt.ylabel("Mean UTH")
plt.title("Daily Mean UTH per Sensor")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#%%
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Choose a day and sensor
day_idx = 0
sensor_idx = 0

# Get lon/lat grids
lon2d, lat2d = np.meshgrid(ref_lon, ref_lat)

fig = plt.figure(figsize=(12,6))
ax = plt.axes(projection=ccrs.Robinson())

im = ax.pcolormesh(
    lon2d,
    lat2d,
    data_4d[day_idx,:,:,sensor_idx],
    transform=ccrs.PlateCarree(),
    cmap="viridis"
)
ax.coastlines()
ax.add_feature(cfeature.BORDERS)
ax.set_title(f"UTH Map - {sensor_names[sensor_idx]} - Day {day_idx}")

cbar = plt.colorbar(im, orientation="horizontal", pad=0.05)
cbar.set_label("UTH")

plt.show()

#%%
for i, name in enumerate(sensor_names):
    std_map = np.nanstd(data_4d[:,:,:,i], axis=0)

    plt.figure(figsize=(10,5))
    plt.imshow(std_map, cmap="magma")
    plt.colorbar(label="STD of UTH")
    plt.title(f"Spatial Variability - {name}")
    plt.xlabel("Longitude pixels")
    plt.ylabel("Latitude pixels")
    plt.tight_layout()
    plt.show()

#%%
for i, name in enumerate(sensor_names):
    vals = data_4d[:,:,:,i]
    print(f"\n{name}")
    print(f"  Mean: {np.nanmean(vals):.2f}")
    print(f"  Std: {np.nanstd(vals):.2f}")
    print(f"  Min: {np.nanmin(vals):.2f}")
    print(f"  Max: {np.nanmax(vals):.2f}")
    print(f"  Missing Fraction: {np.isnan(vals).mean():.3f}")

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
# Make a copy so original data stays intact
data_4d_norm = np.empty_like(data_4d)

for i in range(data_4d.shape[-1]):
    sensor_data = data_4d[..., i]
    data_4d_norm[..., i] = (sensor_data - means[i]) / stds[i]

#%%
for i in range(data_4d.shape[-1]):
    norm_sensor = data_4d_norm[..., i]
    mean_norm = np.nanmean(norm_sensor)
    std_norm = np.nanstd(norm_sensor)
    print(f"Sensor {i}: mean={mean_norm:.3f}, std={std_norm:.3f}")

#%%
def pad_patch(patch, target_shape):
    """Pads a patch (H, W, C) to target_shape if smaller (edge cases)."""
    H, W, C = patch.shape
    padded = np.zeros((target_shape[0], target_shape[1], C), dtype=patch.dtype)
    padded[:H, :W, :] = patch
    return padded

def extract_patches_from_4d(data_4d, patch_size=(64, 64)):
    """
    Splits the full 4D normalized data into non-overlapping patches.
    Returns: list of patches, each of shape (H,W,C)
    """
    patches = []
    T, H, W, C = data_4d.shape
    ph, pw = patch_size

    for t in range(T):
        for i in range(0, H, ph):
            for j in range(0, W, pw):
                patch = data_4d[t, i:i+ph, j:j+pw, :]  # (h,w,c)
                if patch.shape[0] != ph or patch.shape[1] != pw:
                    patch = pad_patch(patch, (ph, pw))
                patches.append(patch)
    return patches

#%%
PATCH_SIZE = (64, 64)  # You can try (32,32) for more data

patches = extract_patches_from_4d(data_4d_norm, patch_size=PATCH_SIZE)
print(f"Total patches extracted: {len(patches)}")
print(f"Patch shape: {patches[0].shape}")

#%%
import torch
from torch.utils.data import Dataset
import random
import numpy as np

class UTHPatchDataset(Dataset):
    def __init__(self, patches, mask_ratio=0.3):
        """
        patches: list of numpy arrays (H,W,C)
        mask_ratio: fraction of pixels to mask randomly
        """
        self.patches = patches
        self.mask_ratio = mask_ratio

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        p = self.patches[idx]  # shape (H,W,C)

        # Convert to tensor
        p_t = torch.tensor(p, dtype=torch.float32)

        # Valid mask: pixels where all channels are finite
        valid_mask = torch.isfinite(p_t).all(dim=-1).float()  # (H,W)

        # Replace NaNs with 0
        p_t[torch.isnan(p_t)] = 0.0

        # Random train mask
        train_mask = torch.ones((p_t.shape[0], p_t.shape[1]), dtype=torch.float32)
        n_pixels = train_mask.numel()
        n_mask = int(self.mask_ratio * n_pixels)
        idxs = random.sample(range(n_pixels), n_mask)
        train_mask.view(-1)[idxs] = 0.0

        # Masked input
        x = p_t.clone()
        x[train_mask == 0] = 0.0

        return (
            x.permute(2,0,1),   # (C,H,W): Input
            p_t.permute(2,0,1), # (C,H,W): Target
            train_mask,         # (H,W)
            valid_mask          # (H,W)
        )

#%%
from sklearn.model_selection import train_test_split

train_patches, val_patches = train_test_split(
    patches, test_size=0.1, random_state=42
)

print(f"Train patches: {len(train_patches)}")
print(f"Val patches: {len(val_patches)}")

#%%
train_dataset = UTHPatchDataset(train_patches, mask_ratio=0.3)
val_dataset = UTHPatchDataset(val_patches, mask_ratio=0.3)

#%%
from torch.utils.data import DataLoader

train_loader = DataLoader(
    train_dataset, batch_size=8, shuffle=True, num_workers=0
)
val_loader = DataLoader(
    val_dataset, batch_size=8, shuffle=False, num_workers=0
)

#%%
batch = next(iter(train_loader))
x, target, train_mask, valid_mask = batch
print("x shape:", x.shape)
print("target shape:", target.shape)
print("train_mask shape:", train_mask.shape)
print("valid_mask shape:", valid_mask.shape)

#%%
import torch
import torch.nn as nn

class AuroraModel(nn.Module):
    def __init__(self, in_channels, embed_dim, num_heads, num_layers, patch_size):
        """
        Aurora-like Transformer model for gap filling and fusion.
        Args:
            in_channels: Number of input channels (4 sensors).
            embed_dim: Embedding dimension (e.g., 256).
            num_heads: Number of transformer attention heads.
            num_layers: Number of transformer layers.
            patch_size: Patch size for embedding.
        """
        super().__init__()

        # Patch embedding layer: Conv2d with stride=patch_size
        self.patch_embed = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

        # Transformer encoder stack
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            activation='gelu',
            batch_first=False  # input shape: (S,B,E)
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # Decoder layer: Upsample back to full resolution
        self.decoder = nn.ConvTranspose2d(
            in_channels=embed_dim,
            out_channels=in_channels,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W)
        Returns:
            reconstructed: (B, C, H, W)
        """
        # Patch embedding
        x = self.patch_embed(x)  # (B, embed_dim, H', W')

        B, C, H, W = x.shape

        # Flatten spatial dimensions and permute for transformer
        x = x.flatten(2).permute(2, 0, 1)  # (S, B, C)

        # Transformer encoding
        x = self.transformer(x)  # (S, B, C)

        # Reshape back to (B, C, H', W')
        x = x.permute(1, 2, 0).reshape(B, C, H, W)

        # Decode to original resolution
        x = self.decoder(x)  # (B, in_channels, H_out, W_out)

        return x

#%%
model = AuroraModel(
    in_channels=4,
    embed_dim=256,    # You can adjust (128/256/512)
    num_heads=4,      # Typical values: 4 or 8
    num_layers=6,     # More layers = more capacity
    patch_size=8      # Can also try 4 or 16
).to(device)

#%%
dummy = torch.rand(2,4,64,64).cuda()
out = model(dummy)
print("Output shape:", out.shape)

#%%
batch = next(iter(train_loader))
inputs, targets, train_mask, valid_mask = [b.to(device) for b in batch]

with torch.no_grad():
    outputs = model(inputs)

print("Input shape :", inputs.shape)
print("Output shape:", outputs.shape)

#%%
def masked_mse_loss(pred, target, train_mask, valid_mask):
    """
    Compute MSE only over pixels that were masked and have valid ground truth.

    Args:
        pred: (B, C, H, W)
        target: (B, C, H, W)
        train_mask: (B, H, W)  -- 1=visible, 0=masked
        valid_mask: (B, H, W)  -- 1=valid data, 0=NaN

    Returns:
        scalar loss
    """
    # Pixels to compute loss on: masked during training & valid
    loss_mask = (train_mask == 0) & (valid_mask == 1)  # (B, H, W)

    # Expand mask across channels
    loss_mask = loss_mask.unsqueeze(1).expand_as(pred)  # (B, C, H, W)

    # Squared error
    sq_error = (pred - target) ** 2

    # Zero out invalid positions
    sq_error = sq_error * loss_mask

    # Compute mean over all valid masked pixels
    return sq_error.sum() / loss_mask.sum().clamp(min=1)

#%%
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

#%%
EPOCHS = 750  # You can adjust as needed

for epoch in range(EPOCHS):
    # Training
    model.train()
    train_losses = []

    for inputs, targets, train_mask, valid_mask in train_loader:
        # Move to GPU
        inputs = inputs.to(device)
        targets = targets.to(device)
        train_mask = train_mask.to(device)
        valid_mask = valid_mask.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)

        loss = masked_mse_loss(outputs, targets, train_mask, valid_mask)
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())

    # Validation
    model.eval()
    val_losses = []

    with torch.no_grad():
        for inputs, targets, train_mask, valid_mask in val_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            train_mask = train_mask.to(device)
            valid_mask = valid_mask.to(device)

            outputs = model(inputs)

            loss = masked_mse_loss(outputs, targets, train_mask, valid_mask)
            val_losses.append(loss.item())

    print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {np.mean(train_losses):.4f} | Val Loss: {np.mean(val_losses):.4f}")

#%%
torch.save(model, "Transformer.pth")
torch.save(model, "Transformer.h5")
#%%
model.eval()

n_samples = 3  # How many patches to visualize

for idx in range(n_samples):
    # Get a sample
    inputs, targets, train_mask, valid_mask = val_dataset[idx]

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
inputs, targets, train_mask, valid_mask = val_dataset[0]

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

    fig, axs = plt.subplots(2, len(sensor_names), figsize=(4*len(sensor_names), 8),
                            subplot_kw={"projection": ccrs.Robinson()})

    for i, sensor in enumerate(sensor_names):
        # Prediction
        ax = axs[0, i]
        ax.set_title(f"Prediction - {sensor}")
        im = ax.imshow(
            p[i],
            transform=ccrs.PlateCarree(),
            extent=[-180,180,-90,90],
            cmap="viridis"
        )
        ax.coastlines()
        ax.add_feature(cfeature.BORDERS, linestyle=':')
        fig.colorbar(im, ax=ax, orientation="vertical", shrink=0.7)

        # Target
        ax = axs[1, i]
        ax.set_title(f"Target - {sensor}")
        im2 = ax.imshow(
            t[i],
            transform=ccrs.PlateCarree(),
            extent=[-180,180,-90,90],
            cmap="viridis"
        )
        ax.coastlines()
        ax.add_feature(cfeature.BORDERS, linestyle=':')
        fig.colorbar(im2, ax=ax, orientation="vertical", shrink=0.7)

    plt.tight_layout()
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

for idx in range(len(val_dataset)):
    inputs, targets, train_mask, valid_mask = val_dataset[idx]

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
