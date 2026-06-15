# -*- coding: utf-8 -*-
"""
Train the 2D MambaFusion model on registered CT-MRI NIfTI volumes and save
model_last/my_cross/fusion_model.pth.

Recommended registered input per case:
  CT:  CT_from_DICOM.nii.gz or CT volume in CT coordinate space
  MRI: MR_resampled_to_CT.nii.gz
  Mask(optional): MR_valid_mask_in_CT.nii.gz

The script converts registered 3D volumes into paired 2D slices, resizes each slice
into 256 x 256, trains the original 2D VSSM_Fusion model slice-by-slice, and saves
fusion_model.pth for subsequent inference.
"""

import os
import sys
import argparse
import random
import time
import datetime
from typing import List, Optional, Tuple

import cv2
import numpy as np
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable

# Make sure this script can import project modules when placed in medicalimagefusion/
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from vmamba_Fusion_efficross import VSSM_Fusion
from loss import Fusionloss


def set_seed(seed: int = 123) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def sitk_to_array(image: sitk.Image) -> np.ndarray:
    """SimpleITK image -> numpy array with shape [z, y, x]."""
    return sitk.GetArrayFromImage(image).astype(np.float32)


def normalize_ct(ct_arr: np.ndarray, lower: float = -300.0, upper: float = 1200.0) -> np.ndarray:
    """CT windowing + min-max normalization to [0, 1]."""
    ct_arr = np.clip(ct_arr.astype(np.float32), lower, upper)
    ct_arr = (ct_arr - lower) / (upper - lower)
    return np.clip(ct_arr, 0.0, 1.0).astype(np.float32)


def normalize_mri(mr_arr: np.ndarray, mask_arr: Optional[np.ndarray] = None, eps: float = 1e-8) -> np.ndarray:
    """MRI z-score normalization followed by rescaling to [0, 1]."""
    mr_arr = mr_arr.astype(np.float32)

    if mask_arr is not None:
        valid = mask_arr > 0
    else:
        valid = np.isfinite(mr_arr) & (mr_arr != 0)
        if valid.sum() < 10:
            valid = np.isfinite(mr_arr)

    values = mr_arr[valid]
    if values.size == 0:
        return np.zeros_like(mr_arr, dtype=np.float32)

    mean = float(values.mean())
    std = float(values.std())
    if std < eps:
        std = 1.0

    z = (mr_arr - mean) / std
    z_valid = z[valid]
    z_min = float(z_valid.min())
    z_max = float(z_valid.max())

    if (z_max - z_min) < eps:
        out = np.zeros_like(z, dtype=np.float32)
    else:
        out = (z - z_min) / (z_max - z_min)

    out = np.clip(out, 0.0, 1.0).astype(np.float32)
    if mask_arr is not None:
        out[~valid] = 0.0
    return out


def resize_slice(slice_2d: np.ndarray, out_size: int = 256, interpolation: int = cv2.INTER_CUBIC) -> np.ndarray:
    return cv2.resize(slice_2d.astype(np.float32), (out_size, out_size), interpolation=interpolation)


def read_train_list(train_list: str) -> List[Tuple[str, str, Optional[str]]]:
    """
    Read a training list file. Each non-empty line supports one of the formats:
      ct_nii,mr_nii
      ct_nii,mr_nii,mask_nii
      ct_nii mr_nii
      ct_nii mr_nii mask_nii
    """
    cases = []
    with open(train_list, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "," in line:
                parts = [p.strip() for p in line.split(",")]
            else:
                parts = line.split()
            if len(parts) not in (2, 3):
                raise ValueError(f"Invalid line {line_no} in {train_list}: {line}")
            ct_path, mr_path = parts[0], parts[1]
            mask_path = parts[2] if len(parts) == 3 and parts[2] else None
            cases.append((ct_path, mr_path, mask_path))
    if len(cases) == 0:
        raise RuntimeError(f"No valid cases found in train list: {train_list}")
    return cases


def prepare_slice_cache(
    cases: List[Tuple[str, str, Optional[str]]],
    cache_dir: str,
    out_size: int = 256,
    ct_lower: float = -300.0,
    ct_upper: float = 1200.0,
    skip_empty_mri: bool = True,
    min_mask_pixels: int = 32,
    force_rebuild: bool = False,
) -> List[str]:
    """
    Convert registered NIfTI volumes into paired 2D slice npz files.
    Each npz stores:
      ct:  [1, H, W], float32 in [0, 1]
      mr:  [1, H, W], float32 in [0, 1]
      name: slice identifier
    """
    ensure_dir(cache_dir)
    index_file = os.path.join(cache_dir, "slice_index.txt")

    if (not force_rebuild) and os.path.exists(index_file):
        with open(index_file, "r", encoding="utf-8") as f:
            cached_files = [line.strip() for line in f if line.strip()]
        cached_files = [p for p in cached_files if os.path.exists(p)]
        if len(cached_files) > 0:
            print(f"Use existing slice cache: {cache_dir}, number of slices: {len(cached_files)}")
            return cached_files

    all_npz_files = []
    print("Preparing paired slice cache...")

    for case_id, (ct_nii, mr_nii, mask_nii) in enumerate(cases):
        print(f"\nCase {case_id}:\n  CT : {ct_nii}\n  MRI: {mr_nii}\n  Mask: {mask_nii}")
        if not os.path.exists(ct_nii):
            raise FileNotFoundError(f"CT file not found: {ct_nii}")
        if not os.path.exists(mr_nii):
            raise FileNotFoundError(f"MRI file not found: {mr_nii}")

        ct_img = sitk.ReadImage(ct_nii)
        mr_img = sitk.ReadImage(mr_nii)

        if ct_img.GetSize() != mr_img.GetSize():
            raise ValueError(
                f"CT and MRI sizes are different. CT={ct_img.GetSize()}, MRI={mr_img.GetSize()}. "
                "Please use MR_resampled_to_CT.nii.gz or resample MRI to CT space before training."
            )

        ct_raw = sitk_to_array(ct_img)  # [z, y, x]
        mr_raw = sitk_to_array(mr_img)

        mask_arr = None
        if mask_nii is not None and os.path.exists(mask_nii):
            mask_img = sitk.ReadImage(mask_nii)
            if mask_img.GetSize() != ct_img.GetSize():
                raise ValueError(f"Mask size {mask_img.GetSize()} does not match CT size {ct_img.GetSize()}.")
            mask_arr = sitk_to_array(mask_img) > 0

        ct_arr = normalize_ct(ct_raw, lower=ct_lower, upper=ct_upper)
        mr_arr = normalize_mri(mr_raw, mask_arr=mask_arr)

        z_num = ct_arr.shape[0]
        for k in range(z_num):
            if skip_empty_mri:
                if mask_arr is not None:
                    if int(mask_arr[k].sum()) < min_mask_pixels:
                        continue
                else:
                    if int((mr_arr[k] > 1e-6).sum()) < min_mask_pixels:
                        continue

            ct_slice = resize_slice(ct_arr[k], out_size=out_size, interpolation=cv2.INTER_CUBIC)
            mr_slice = resize_slice(mr_arr[k], out_size=out_size, interpolation=cv2.INTER_CUBIC)

            ct_slice = ct_slice[None, :, :].astype(np.float32)
            mr_slice = mr_slice[None, :, :].astype(np.float32)

            out_name = f"case{case_id:04d}_slice{k:04d}.npz"
            out_path = os.path.join(cache_dir, out_name)
            np.savez_compressed(out_path, ct=ct_slice, mr=mr_slice, name=out_name)
            all_npz_files.append(out_path)

        print(f"  Added slices so far: {len(all_npz_files)}")

    if len(all_npz_files) == 0:
        raise RuntimeError("No training slices were generated. Check MRI mask or input volumes.")

    with open(index_file, "w", encoding="utf-8") as f:
        for p in all_npz_files:
            f.write(p + "\n")

    print(f"\nSaved slice cache index: {index_file}")
    print(f"Total training slices: {len(all_npz_files)}")
    return all_npz_files


class RegisteredSliceDataset(Dataset):
    def __init__(self, npz_files: List[str]):
        self.npz_files = npz_files

    def __len__(self):
        return len(self.npz_files)

    def __getitem__(self, index):
        npz_path = self.npz_files[index]
        data = np.load(npz_path, allow_pickle=True)
        ct = torch.from_numpy(data["ct"].astype(np.float32))  # [1, H, W]
        mr = torch.from_numpy(data["mr"].astype(np.float32))  # [1, H, W]
        name = str(data["name"])
        return ct, mr, name


def train_fusion(args):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu >= 0 else "cpu")
    if device.type != "cuda":
        raise RuntimeError("This training script uses the original Fusionloss, which requires CUDA. Please run with a CUDA GPU.")
    torch.cuda.set_device(args.gpu)
    print(f"Using device: {device}")

    set_seed(args.seed)

    cases = read_train_list(args.train_list)
    npz_files = prepare_slice_cache(
        cases=cases,
        cache_dir=args.cache_dir,
        out_size=args.out_size,
        ct_lower=args.ct_lower,
        ct_upper=args.ct_upper,
        skip_empty_mri=not args.keep_empty_slices,
        min_mask_pixels=args.min_mask_pixels,
        force_rebuild=args.force_rebuild_cache,
    )

    dataset = RegisteredSliceDataset(npz_files)
    train_loader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    print(f"Training slices: {len(dataset)}")
    print(f"Batch size: {args.batch_size}; iterations per epoch: {len(train_loader)}")

    model = VSSM_Fusion(in_chans=1).to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = Fusionloss()

    save_dir = args.save_dir
    ensure_dir(save_dir)
    final_weight_path = os.path.join(save_dir, "fusion_model.pth")
    last_weight_path = os.path.join(save_dir, "fusion_model_last.pth")

    global_step = 0
    best_epoch_loss = float("inf")
    start_time = time.time()

    for epoch in range(args.epochs):
        model.train()

        if args.lr_decay > 0 and args.lr_decay != 1.0:
            lr_this_epoch = args.lr * (args.lr_decay ** epoch)
            for pg in optimizer.param_groups:
                pg["lr"] = lr_this_epoch
        else:
            lr_this_epoch = args.lr

        epoch_loss = 0.0
        epoch_loss_in = 0.0
        epoch_loss_ssim = 0.0
        epoch_loss_grad = 0.0

        for it, (ct, mr, name) in enumerate(train_loader):
            ct = Variable(ct).to(device, non_blocking=True)  # [B, 1, 256, 256]
            mr = Variable(mr).to(device, non_blocking=True)

            fused = model(ct, mr)
            fused = torch.clamp(fused, 0.0, 1.0)

            loss_total, loss_in, ssim_value, loss_grad = criterion(
                image_vis=ct,
                image_ir=mr,
                generate_img=fused,
                i=0,
                labels=None,
            )

            optimizer.zero_grad(set_to_none=True)
            loss_total.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            global_step += 1
            epoch_loss += float(loss_total.item())
            epoch_loss_in += float(loss_in.item())
            epoch_loss_ssim += float(ssim_value.item())
            epoch_loss_grad += float(loss_grad.item())

            if global_step % args.log_interval == 0:
                elapsed = time.time() - start_time
                done_steps = epoch * len(train_loader) + it + 1
                total_steps = args.epochs * len(train_loader)
                eta_seconds = int(elapsed / max(done_steps, 1) * (total_steps - done_steps))
                eta = str(datetime.timedelta(seconds=eta_seconds))
                print(
                    f"Epoch [{epoch + 1}/{args.epochs}] "
                    f"Iter [{it + 1}/{len(train_loader)}] "
                    f"Step {global_step} | "
                    f"lr={lr_this_epoch:.6g} | "
                    f"loss={loss_total.item():.4f} | "
                    f"loss_in={loss_in.item():.4f} | "
                    f"ssim_loss={ssim_value.item():.4f} | "
                    f"loss_grad={loss_grad.item():.4f} | "
                    f"eta={eta}"
                )

        n_iter = max(len(train_loader), 1)
        epoch_loss /= n_iter
        epoch_loss_in /= n_iter
        epoch_loss_ssim /= n_iter
        epoch_loss_grad /= n_iter

        print(
            f"\nEpoch {epoch + 1}/{args.epochs} finished | "
            f"avg_loss={epoch_loss:.4f}, avg_loss_in={epoch_loss_in:.4f}, "
            f"avg_ssim_loss={epoch_loss_ssim:.4f}, avg_loss_grad={epoch_loss_grad:.4f}\n"
        )

        # Save last checkpoint every epoch.
        torch.save(model.state_dict(), last_weight_path)

        # Save best checkpoint according to training loss.
        if epoch_loss < best_epoch_loss:
            best_epoch_loss = epoch_loss
            torch.save(model.state_dict(), final_weight_path)
            print(f"Saved best model to: {final_weight_path}")

        if args.save_epoch_checkpoints:
            epoch_path = os.path.join(save_dir, f"fusion_model_epoch_{epoch + 1:03d}.pth")
            torch.save(model.state_dict(), epoch_path)

    # Make sure fusion_model.pth exists after training.
    if not os.path.exists(final_weight_path):
        torch.save(model.state_dict(), final_weight_path)

    print(f"Training done. Best training loss: {best_epoch_loss:.4f}")
    print(f"Final model path: {final_weight_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train MambaFusion on registered CT-MRI NIfTI volumes.")
    parser.add_argument("--train_list", required=True, help="TXT file. Each line: ct_nii,mr_nii[,mask_nii]")
    parser.add_argument("--cache_dir", default="registered_slice_cache", help="Directory to store preprocessed paired slice npz files")
    parser.add_argument("--save_dir", default=os.path.join("model_last", "my_cross"), help="Directory to save fusion_model.pth")

    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Initial learning rate")
    parser.add_argument("--lr_decay", type=float, default=0.75, help="Per-epoch LR decay. Set 1.0 to disable")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Adam weight decay")
    parser.add_argument("--grad_clip", type=float, default=0.0, help="Gradient clipping max norm. Set 0 to disable")

    parser.add_argument("--out_size", type=int, default=256, help="2D slice size fed into the model")
    parser.add_argument("--ct_lower", type=float, default=-300.0, help="Lower CT window bound")
    parser.add_argument("--ct_upper", type=float, default=1200.0, help="Upper CT window bound")
    parser.add_argument("--keep_empty_slices", action="store_true", help="Keep slices with nearly empty MRI signal/mask")
    parser.add_argument("--min_mask_pixels", type=int, default=32, help="Minimum valid MRI mask pixels to keep a slice")
    parser.add_argument("--force_rebuild_cache", action="store_true", help="Rebuild slice cache even if it already exists")

    parser.add_argument("--gpu", type=int, default=0, help="GPU id")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers. Use 0 on Windows if multiprocessing has issues")
    parser.add_argument("--seed", type=int, default=123, help="Random seed")
    parser.add_argument("--log_interval", type=int, default=10, help="Print log every N steps")
    parser.add_argument("--save_epoch_checkpoints", action="store_true", help="Save checkpoint at each epoch")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_fusion(args)
