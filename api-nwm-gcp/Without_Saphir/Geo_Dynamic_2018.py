#%%
import os
import numpy as np
import xarray as xr

# Base directory
base_dir = r"/MICROWAVE_UTH_DATA_NOAA"

# Paths to each sensor's folder
sensor_dirs = {
    "Metop_A": os.path.join(base_dir, "MHS_Metop_A_2007_2021"),
    "Metop_B": os.path.join(base_dir, "MHS_Metop_B_2013_2021"),
    "NOAA_18": os.path.join(base_dir, "MHS_NOAA_18_2006_2018"),
    "NOAA_19": os.path.join(base_dir, "MHS_NOAA_19_2016_2021"),
}

# List of all .nc files per sensor
sensor_files = {
    k: sorted([os.path.join(v, f) for f in os.listdir(v) if f.endswith(".nc")])
    for k, v in sensor_dirs.items()
}

# Helper function to load one sensor stack
def load_sensor_stack(file_list, variable="uth_mean_ascend_descend"):
    uth_arrays = []
    for f in file_list:
        ds = xr.open_dataset(f)
        uth = ds[variable]
        # Remove time dimension if singleton
        if "time" in uth.dims and uth.sizes["time"] == 1:
            uth = uth.squeeze("time")
        uth_arrays.append(uth)
        ds.close()
    return xr.concat(uth_arrays, dim="time")

# Load all sensors
sensor_data = {s: load_sensor_stack(files) for s, files in sensor_files.items()}

# Reference lat/lon grid
ref_lat = sensor_data["Metop_A"]["lat"]
ref_lon = sensor_data["Metop_A"]["lon"]

# Align all data to the reference grid
sensor_data_aligned = {
    s: da.interp(lat=ref_lat, lon=ref_lon, method="nearest")
    for s, da in sensor_data.items()
}

# Convert to numpy arrays
arrays = [sensor_data_aligned[s].values for s in sensor_dirs.keys()]

# Stack to 4D array: (time, lat, lon, sensor)
data_4d = np.stack(arrays, axis=-1)

print("Shape of stacked data:", data_4d.shape)
print("NaN proportion:", np.isnan(data_4d).mean())

# Compute mean/std per sensor (ignoring NaNs)
means = []
stds = []
for i in range(data_4d.shape[-1]):
    vals = data_4d[..., i]
    means.append(np.nanmean(vals))
    stds.append(np.nanstd(vals))

# Normalize
data_4d_norm = (data_4d - means) / stds

print("Normalization complete.")

#%%
import torch
from torch.utils.data import Dataset

class FullUTHDataset(Dataset):
    def __init__(self, data, mask_ratio=0.3):
        """
        data: 4D numpy array of shape (T, H, W, C)
        mask_ratio: proportion of valid pixels to mask during training
        """
        self.data = data
        self.mask_ratio = mask_ratio
        self.time_dim = data.shape[0]

    def __len__(self):
        return self.time_dim

    def __getitem__(self, idx):
        sample = self.data[idx]  # shape (H, W, C)
        sample_tensor = torch.tensor(sample, dtype=torch.float32)  # (H, W, C)

        # Build valid mask (1 for valid, 0 for NaNs)
        valid_mask = torch.isfinite(sample_tensor).all(dim=-1).float()  # shape (H, W)

        # Replace NaNs with 0.0 (for tensor safety)
        sample_tensor[~torch.isfinite(sample_tensor)] = 0.0

        # Build training mask (1 = visible, 0 = mask out)
        train_mask = valid_mask.clone()
        n_pixels = train_mask.sum().item()
        n_mask = int(self.mask_ratio * n_pixels)

        if n_mask > 0:
            valid_indices = torch.nonzero(train_mask, as_tuple=False)
            mask_indices = valid_indices[torch.randperm(len(valid_indices))[:n_mask]]
            train_mask[mask_indices[:, 0], mask_indices[:, 1]] = 0.0

        # Masked input: set masked pixels to 0
        masked_input = sample_tensor.clone()
        for c in range(sample_tensor.shape[-1]):
            masked_input[:, :, c][train_mask == 0] = 0.0

        # Format as (C, H, W)
        input_tensor = masked_input.permute(2, 0, 1)
        target_tensor = sample_tensor.permute(2, 0, 1)
        train_mask_tensor = train_mask  # shape (H, W)
        valid_mask_tensor = valid_mask  # shape (H, W)

        return input_tensor, target_tensor, train_mask_tensor, valid_mask_tensor

#%%
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

# Split the full 4D dataset (time-based split)
train_data, val_data = train_test_split(data_4d_norm, test_size=0.1, random_state=42)

# Dataset objects
train_ds = FullUTHDataset(train_data, mask_ratio=0.3)
val_ds = FullUTHDataset(val_data, mask_ratio=0.3)

# DataLoaders
train_loader = DataLoader(train_ds, batch_size=1, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=1)

#%%
import torch
import torch.nn as nn
import torch.nn.functional as F

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
        self.conv0_4 = conv_block(filters * 4, filters)

        self.final = nn.Conv2d(filters, out_channels, kernel_size=1)

    def upsample_to(self, src, target):
        """Upsample src to the spatial size of target."""
        return F.interpolate(src, size=target.shape[2:], mode="bilinear", align_corners=True)

    def forward(self, x):
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        x3_1 = self.conv3_1(torch.cat([x3_0, self.upsample_to(x4_0, x3_0)], dim=1))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.upsample_to(x3_0, x2_0)], dim=1))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.upsample_to(x2_0, x1_0)], dim=1))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.upsample_to(x1_0, x0_0)], dim=1))

        x2_2 = self.conv2_2(torch.cat([x2_0, self.upsample_to(x3_1, x2_0)], dim=1))
        x1_2 = self.conv1_2(torch.cat([x1_0, self.upsample_to(x2_1, x1_0)], dim=1))
        x0_2 = self.conv0_2(torch.cat([x0_0, self.upsample_to(x1_1, x0_0)], dim=1))

        x1_3 = self.conv1_3(torch.cat([
            x1_0, x1_1, self.upsample_to(x2_2, x1_0)
        ], dim=1))
        x0_3 = self.conv0_3(torch.cat([
            x0_0, x0_1, self.upsample_to(x1_2, x0_0)
        ], dim=1))

        x0_4 = self.conv0_4(torch.cat([
            x0_0, x0_1, x0_2, x0_3
        ], dim=1))

        return self.final(x0_4)

#%%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNetPP(in_channels=4, out_channels=4).to(device)

#%%
import torch
import torch.nn.functional as F

# Masked Mean Absolute Error
def masked_mae(pred, target, valid_mask):
    """
    pred: (B, C, H, W)
    target: (B, C, H, W)
    valid_mask: (B, 1, H, W) where 1 = valid pixel, 0 = ignore (NaNs)
    """
    # Broadcast mask to channels
    loss_mask = valid_mask.expand_as(pred)
    abs_err = torch.abs(pred - target) * loss_mask
    return abs_err.sum() / loss_mask.sum().clamp(min=1)

# Masked Mean Squared Error
def masked_mse(pred, target, valid_mask):
    loss_mask = valid_mask.expand_as(pred)
    sq_err = (pred - target) ** 2 * loss_mask
    return sq_err.sum() / loss_mask.sum().clamp(min=1)

# Masked Root Mean Squared Error
def masked_rmse(pred, target, valid_mask):
    return torch.sqrt(masked_mse(pred, target, valid_mask))

# Optionally: Masked SSIM (needs pytorch_msssim installed)
try:
    from pytorch_msssim import ssim
    def masked_ssim(pred, target, valid_mask):
        # Note: pytorch_msssim does not natively support masking
        # You could set non-valid pixels to target values to avoid penalizing them
        pred_masked = pred.clone()
        target_masked = target.clone()
        pred_masked[valid_mask.expand_as(pred)==0] = target[valid_mask.expand_as(pred)==0]
        return ssim(pred_masked, target_masked, data_range=1.0, size_average=True)
except ImportError:
    print("pytorch_msssim not installed, skipping SSIM.")
    ssim = None

# Example combined loss (optional)
def masked_mae_ssim(pred, target, valid_mask, alpha=0.8):
    mae = masked_mae(pred, target, valid_mask)
    if ssim is not None:
        ssim_loss = 1 - masked_ssim(pred, target, valid_mask)
        return alpha * mae + (1 - alpha) * ssim_loss
    else:
        return mae

#%%
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

#%%
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np

# Split indices
num_samples = data_4d_norm.shape[0]
indices = np.arange(num_samples)
train_idx, val_idx = train_test_split(indices, test_size=0.1, random_state=42)

# Dataset class
class FullImageDataset(Dataset):
    def __init__(self, data, indices):
        """
        data: normalized data (T, H, W, C)
        indices: list of time indices
        """
        self.data = data
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        t = self.indices[idx]
        x = self.data[t]  # (H, W, C)

        # Build valid mask (1 = valid, 0 = nan)
        valid_mask = np.isfinite(x).all(axis=-1).astype(np.float32)  # (H, W)

        # Fill NaNs with 0s (or other value)
        x_filled = np.nan_to_num(x, nan=0.0)

        # Convert to torch tensors
        x_tensor = torch.tensor(x_filled, dtype=torch.float32).permute(2,0,1)  # (C,H,W)
        valid_mask_tensor = torch.tensor(valid_mask, dtype=torch.float32).unsqueeze(0)  # (1,H,W)

        return x_tensor, x_tensor, valid_mask_tensor
        # Note: target is same as input (denoising/filling task)

# Datasets
train_dataset = FullImageDataset(data_4d_norm, train_idx)
val_dataset = FullImageDataset(data_4d_norm, val_idx)

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)

print(f"Train samples: {len(train_dataset)}")
print(f"Val samples: {len(val_dataset)}")

#%%
import time

NUM_EPOCHS = 750

train_losses = []
val_losses = []
val_maes = []
val_rmses = []
val_mses = []

for epoch in range(NUM_EPOCHS):
    start_time = time.time()

    # ---------------- Training ----------------
    model.train()
    train_loss_epoch = 0.0
    n_train_batches = 0

    for xb, yb, valid_maskb in train_loader:
        xb, yb, valid_maskb = xb.to(device), yb.to(device), valid_maskb.to(device)

        optimizer.zero_grad()
        preds = model(xb)
        loss = masked_mae(preds, yb, valid_maskb)
        loss.backward()
        optimizer.step()

        train_loss_epoch += loss.item()
        n_train_batches += 1

    mean_train_loss = train_loss_epoch / n_train_batches
    train_losses.append(mean_train_loss)

    # ---------------- Validation ----------------
    model.eval()
    val_loss_epoch = 0.0
    val_mae_epoch = 0.0
    val_rmse_epoch = 0.0
    val_mse_epoch = 0.0
    n_val_batches = 0

    with torch.no_grad():
        for xb, yb, valid_maskb in val_loader:
            xb, yb, valid_maskb = xb.to(device), yb.to(device), valid_maskb.to(device)

            preds = model(xb)

            loss = masked_mae(preds, yb, valid_maskb)
            mae = masked_mae(preds, yb, valid_maskb)
            mse = masked_mse(preds, yb, valid_maskb)
            rmse = masked_rmse(preds, yb, valid_maskb)

            val_loss_epoch += loss.item()
            val_mae_epoch += mae.item()
            val_mse_epoch += mse.item()
            val_rmse_epoch += rmse.item()
            n_val_batches += 1

    mean_val_loss = val_loss_epoch / n_val_batches
    mean_val_mae = val_mae_epoch / n_val_batches
    mean_val_mse = val_mse_epoch / n_val_batches
    mean_val_rmse = val_rmse_epoch / n_val_batches

    val_losses.append(mean_val_loss)
    val_maes.append(mean_val_mae)
    val_mses.append(mean_val_mse)
    val_rmses.append(mean_val_rmse)

    elapsed = time.time() - start_time
    print(
        f"Epoch {epoch+1:03d}/{NUM_EPOCHS} — "
        f"Train Loss: {mean_train_loss:.5f} | Val Loss: {mean_val_loss:.5f} | "
        f"Val MAE: {mean_val_mae:.4f} | Val RMSE: {mean_val_rmse:.4f} | Time: {elapsed:.1f}s"
    )

#%%
torch.save(model.state_dict(), "UNetPP_GEO_AUG2018_model.pth")

#%%
import numpy as np
from skimage.metrics import structural_similarity as ssim_metric
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from sklearn.metrics import mean_squared_error, mean_absolute_error

mse_list = []
rmse_list = []
mae_list = []
ssim_list = []
psnr_list = []

model.eval()

with torch.no_grad():
    for xb, yb, valid_maskb in val_loader:
        xb = xb.to(device)
        preds = model(xb).cpu()
        targets = yb.cpu()
        valid_masks = valid_maskb.cpu()

        B, C, H, W = preds.shape

        # If mask has shape (B,1,H,W), squeeze dim=1
        if valid_masks.dim() == 4:
            valid_masks = valid_masks.squeeze(1)  # Now (B,H,W)

        for i in range(B):
            pred_np = preds[i].numpy()       # (C,H,W)
            target_np = targets[i].numpy()   # (C,H,W)
            mask_np = valid_masks[i].numpy() # (H,W)

            for ch in range(C):
                pred_ch = pred_np[ch]
                target_ch = target_np[ch]

                eval_mask = mask_np.astype(bool)

                if eval_mask.sum() == 0:
                    continue

                pred_valid = pred_ch[eval_mask]
                target_valid = target_ch[eval_mask]

                mse = mean_squared_error(target_valid, pred_valid)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(target_valid, pred_valid)

                # SSIM & PSNR on reconstructed arrays
                pred_full = pred_ch.copy()
                target_full = target_ch.copy()
                pred_full[~eval_mask] = target_full[~eval_mask]

                ssim_val = ssim_metric(
                    target_full,
                    pred_full,
                    data_range=target_full.max() - target_full.min()
                )
                psnr_val = psnr_metric(
                    target_full,
                    pred_full,
                    data_range=target_full.max() - target_full.min()
                )

                mse_list.append(mse)
                rmse_list.append(rmse)
                mae_list.append(mae)
                ssim_list.append(ssim_val)
                psnr_list.append(psnr_val)

print("\n=== Validation Metrics (Valid Regions) ===")
print(f"Mean MSE:   {np.mean(mse_list):.4f}")
print(f"Mean RMSE:  {np.mean(rmse_list):.4f}")
print(f"Mean MAE:   {np.mean(mae_list):.4f}")
print(f"Mean SSIM:  {np.mean(ssim_list):.4f}")
print(f"Mean PSNR:  {np.mean(psnr_list):.2f} dB")

#%%
import numpy as np

means = []
stds = []

for i in range(data_4d.shape[-1]):
    sensor_data = data_4d[..., i]
    mean_val = np.nanmean(sensor_data)
    std_val = np.nanstd(sensor_data)
    means.append(mean_val)
    stds.append(std_val)

means = np.array(means)
stds = np.array(stds)

print("Sensor-wise Means:", means)
print("Sensor-wise Stds:", stds)

#%%
mean_rmse_norm = 0.0363
mean_mae_norm = 0.0181

stds = [18.108423, 18.000591, 18.094555, 15.177217]
sensors = ["Metop_A", "Metop_B", "NOAA_18", "NOAA_19"]

print("==== Approximate Error per Sensor (in % UTH) ====")
for sensor, std in zip(sensors, stds):
    rmse_real = mean_rmse_norm * std
    mae_real = mean_mae_norm * std
    print(f"{sensor}: RMSE = {rmse_real:.2f}% UTH | MAE = {mae_real:.2f}% UTH")

#%%
# Make sure your predictions and targets are on CPU
predictions = preds.cpu()
targets = targets.cpu()

# Convert means and stds to torch tensors, same dtype and device
means_tensor = torch.tensor([36.98, 36.66, 36.63, 34.18], dtype=predictions.dtype, device=predictions.device)
stds_tensor = torch.tensor([18.11, 18.00, 18.09, 15.18], dtype=predictions.dtype, device=predictions.device)

# Denormalize
predictions_denorm = predictions * stds_tensor.view(1, -1, 1, 1) + means_tensor.view(1, -1, 1, 1)
targets_denorm = targets * stds_tensor.view(1, -1, 1, 1) + means_tensor.view(1, -1, 1, 1)

print("Done denormalizing.")

#%%
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np

def plot_robinson_comparison(predictions, targets, valid_masks, sensor_names, time_idx=0):
    """
    predictions: (T, C, H, W) tensor or array
    targets:     (T, C, H, W) tensor or array
    valid_masks: (T, H, W) tensor or array
    time_idx:    which timestep to visualize
    """

    preds_np = predictions[time_idx].cpu().numpy()
    targets_np = targets[time_idx].cpu().numpy()
    mask_np = valid_masks[time_idx].cpu().numpy()

    n_sensors = preds_np.shape[0]

    fig, axs = plt.subplots(
        2, n_sensors,
        figsize=(5 * n_sensors, 8),
        subplot_kw={"projection": ccrs.Robinson()}
    )

    axs = np.atleast_2d(axs)

    # Common color scale across all sensors
    vmin = min(targets_np.min(), preds_np.min())
    vmax = max(targets_np.max(), preds_np.max())

    for i, sensor in enumerate(sensor_names):
        # Ground truth
        ax_t = axs[0, i]
        ax_t.set_title(f"Target - {sensor}")
        im = ax_t.imshow(
            targets_np[i],
            transform=ccrs.PlateCarree(),
            extent=[-180, 180, -90, 90],
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
        )
        ax_t.coastlines()
        ax_t.add_feature(cfeature.BORDERS, linestyle=":")

        # Prediction
        ax_p = axs[1, i]
        ax_p.set_title(f"Prediction - {sensor}")
        ax_p.imshow(
            preds_np[i],
            transform=ccrs.PlateCarree(),
            extent=[-180, 180, -90, 90],
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
        )
        ax_p.coastlines()
        ax_p.add_feature(cfeature.BORDERS, linestyle=":")

    # Single colorbar for all images
    cbar_ax = fig.add_axes([0.2, 0.05, 0.6, 0.02])
    fig.colorbar(im, cax=cbar_ax, orientation="horizontal", label="Normalized UTH")

    plt.tight_layout()
    plt.show()

#%%


plot_robinson_comparison(
    predictions_denorm,
    targets_denorm,
    valid_masks,
    sensor_names=["Metop_A", "Metop_B", "NOAA_18", "NOAA_19"],
    time_idx=0  # Change this to visualize different timesteps
)

#%%
import torch
import numpy as np

# Ensure model in eval mode
model.eval()

# Precompute normalization tensors
means_tensor = torch.tensor(means, dtype=torch.float32).view(1, -1, 1, 1)
stds_tensor = torch.tensor(stds, dtype=torch.float32).view(1, -1, 1, 1)

# Prepare storage
reconstructed = np.zeros_like(data_4d_norm, dtype=np.float32)

# Loop over time slices
for t in range(data_4d_norm.shape[0]):
    print(f"Reconstructing timestep {t+1}/{data_4d_norm.shape[0]}...")

    # (H, W, C)
    data_slice = data_4d_norm[t]

    # Build mask of valid pixels
    valid_mask = np.isfinite(data_slice).all(axis=-1).astype(np.float32)

    # Fill NaNs with 0
    data_slice[np.isnan(data_slice)] = 0.0

    # Convert to torch tensor (C,H,W)
    input_tensor = torch.tensor(data_slice.transpose(2,0,1)).unsqueeze(0).to(device)
    valid_tensor = torch.tensor(valid_mask).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(input_tensor).cpu().squeeze(0)  # (C,H,W)

    # Convert prediction to numpy
    pred_np = pred.numpy().transpose(1,2,0)  # (H,W,C)

    # Fill missing pixels
    reconstructed_slice = data_slice.copy()
    missing = valid_mask == 0
    reconstructed_slice[missing] = pred_np[missing]

    # Store reconstructed slice
    reconstructed[t] = reconstructed_slice

#%%
# Denormalize: X_denorm = X_norm * std + mean
reconstructed_denorm = reconstructed * stds + means

#%%
import xarray as xr

# Create DataArray
recon_da = xr.DataArray(
    reconstructed_denorm,
    dims=("time", "lat", "lon", "sensor"),
    coords={
        "time": np.arange(data_4d_norm.shape[0]),
        "lat": ref_lat,
        "lon": ref_lon,
        "sensor": ["Metop_A", "Metop_B", "NOAA_18", "NOAA_19"]
    },
    name="uth_filled"
)

# Save to NetCDF
recon_da.to_netcdf("reconstructed_uth_gap_filled.nc")
print("Saved to reconstructed_uth_gap_filled.nc")

#%%
import matplotlib.pyplot as plt
plt.imshow(reconstructed_denorm[0,:,:,0], cmap="viridis")
plt.title("Reconstructed NOAA_19 time=0")
plt.colorbar(label="% UTH")

#%%
import numpy as np
import xarray as xr

# Example: reconstructed_denorm = np.array(...)  # (T, H, W, 4)
# Make sure it exists in your workspace.


reconstructed_fused = np.nanmean(reconstructed_denorm, axis=-1)  # Shape: (T, H, W)

print("Fused shape:", reconstructed_fused.shape)


fused_da = xr.DataArray(
    reconstructed_fused,
    dims=("time", "lat", "lon"),
    coords={
        "time": np.arange(reconstructed_fused.shape[0]),
        "lat": ref_lat,
        "lon": ref_lon
    },
    name="uth_fused"
)


output_path = "reconstructed_fused_uth.nc"
fused_da.to_netcdf(output_path)
print(f"Saved fused reconstruction to {output_path}")




#%%
import matplotlib.pyplot as plt

i = 30  # Index of time step to visualize

plt.figure(figsize=(10, 5))
plt.imshow(reconstructed_fused[i], cmap="viridis", origin="lower")
plt.title(f"Fused Reconstruction - Time Step {i}")
plt.colorbar(label="% UTH")
plt.xlabel("Longitude Index")
plt.ylabel("Latitude Index")
plt.show()