# -*- coding: utf-8 -*-
"""
Slice-by-slice inference for registered CT-MRI volumes using the trained 2D MambaFusion model.

Recommended input after registration:
  CT:  CT_from_DICOM.nii.gz or the CT volume in CT space
  MRI: MR_resampled_to_CT.nii.gz
  Mask (optional): MR_valid_mask_in_CT.nii.gz

The script resizes each paired CT/MRI slice to 256 x 256, performs 2D fusion,
and saves a 3D fused NIfTI volume with size 256 x 256 x n.
"""

import os
import sys
import argparse
import numpy as np
import cv2
import SimpleITK as sitk
import torch

# Make sure the project root can be imported when this script is placed in medicalimagefusion/
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from vmamba_Fusion_efficross import VSSM_Fusion


def strip_module_prefix(state_dict):
    """Remove 'module.' prefix if the model was saved with DataParallel."""
    new_state = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k[len("module."):]
        new_state[k] = v
    return new_state


def load_model(weights_path, device):
    model = VSSM_Fusion(in_chans=1).to(device)
    state = torch.load(weights_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    state = strip_module_prefix(state)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def sitk_to_array(image):
    """SimpleITK image -> numpy array with shape [z, y, x]."""
    return sitk.GetArrayFromImage(image).astype(np.float32)


def normalize_ct(ct_arr, lower=-300.0, upper=1200.0):
    ct_arr = np.clip(ct_arr, lower, upper)
    ct_arr = (ct_arr - lower) / (upper - lower)
    return np.clip(ct_arr, 0.0, 1.0).astype(np.float32)


def normalize_mri(mr_arr, mask_arr=None, eps=1e-8):
    """
    MRI normalization: z-score normalization followed by rescaling to [0, 1].
    If a valid MRI mask is provided, statistics and rescaling are computed inside the mask.
    """
    mr_arr = mr_arr.astype(np.float32)
    if mask_arr is not None:
        valid = mask_arr > 0
    else:
        # Avoid letting large zero background dominate if the resampled MRI has empty regions.
        valid = np.isfinite(mr_arr) & (mr_arr != 0)
        if valid.sum() < 10:
            valid = np.isfinite(mr_arr)

    values = mr_arr[valid]
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


def resize_slice(slice_2d, out_size=256, interpolation=cv2.INTER_CUBIC):
    return cv2.resize(slice_2d.astype(np.float32), (out_size, out_size), interpolation=interpolation)


def save_png_slice(image_2d_float, output_path):
    image_uint8 = np.clip(image_2d_float * 255.0, 0, 255).astype(np.uint8)
    cv2.imwrite(output_path, image_uint8)


def make_output_like_ct_256(fused_arr_zyx, ct_img, out_size=256):
    """
    Create SimpleITK image from fused array [z, 256, 256].
    The in-plane spacing is adjusted to preserve the CT physical field of view.
    """
    out_img = sitk.GetImageFromArray(fused_arr_zyx.astype(np.float32))

    ct_size = ct_img.GetSize()       # (x, y, z)
    ct_spacing = ct_img.GetSpacing() # (sx, sy, sz)
    new_spacing = (
        ct_spacing[0] * ct_size[0] / float(out_size),
        ct_spacing[1] * ct_size[1] / float(out_size),
        ct_spacing[2],
    )
    out_img.SetSpacing(new_spacing)
    out_img.SetOrigin(ct_img.GetOrigin())
    out_img.SetDirection(ct_img.GetDirection())
    return out_img


def fuse_registered_volumes(
    ct_nii,
    mr_nii,
    weights,
    out_nii,
    mask_nii=None,
    out_png_dir=None,
    out_size=256,
    ct_lower=-300.0,
    ct_upper=1200.0,
    gpu=0,
    keep_ct_outside_mask=True,
):
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() and gpu >= 0 else "cpu")
    print(f"Using device: {device}")

    ct_img = sitk.ReadImage(ct_nii)
    mr_img = sitk.ReadImage(mr_nii)

    if ct_img.GetSize() != mr_img.GetSize():
        raise ValueError(f"CT and MRI sizes are different: CT={ct_img.GetSize()}, MRI={mr_img.GetSize()}. "
                         f"Use MR_resampled_to_CT.nii.gz or resample MRI to CT space first.")

    ct_arr_raw = sitk_to_array(ct_img)  # [z, y, x]
    mr_arr_raw = sitk_to_array(mr_img)

    mask_arr = None
    if mask_nii is not None and os.path.exists(mask_nii):
        mask_img = sitk.ReadImage(mask_nii)
        if mask_img.GetSize() != ct_img.GetSize():
            raise ValueError(f"Mask size {mask_img.GetSize()} does not match CT size {ct_img.GetSize()}.")
        mask_arr = sitk_to_array(mask_img) > 0

    ct_arr = normalize_ct(ct_arr_raw, lower=ct_lower, upper=ct_upper)
    mr_arr = normalize_mri(mr_arr_raw, mask_arr=mask_arr)

    model = load_model(weights, device)

    z_num = ct_arr.shape[0]
    fused_slices = []

    if out_png_dir is not None:
        os.makedirs(out_png_dir, exist_ok=True)

    for k in range(z_num):
        ct_slice = resize_slice(ct_arr[k], out_size=out_size, interpolation=cv2.INTER_CUBIC)
        mr_slice = resize_slice(mr_arr[k], out_size=out_size, interpolation=cv2.INTER_CUBIC)

        ct_tensor = torch.from_numpy(ct_slice[None, None, :, :]).float().to(device)
        mr_tensor = torch.from_numpy(mr_slice[None, None, :, :]).float().to(device)

        with torch.no_grad():
            fused = model(ct_tensor, mr_tensor)
            fused = torch.clamp(fused, 0.0, 1.0)

        fused_np = fused.squeeze().detach().cpu().numpy().astype(np.float32)

        if mask_arr is not None and keep_ct_outside_mask:
            mask_slice = resize_slice(mask_arr[k].astype(np.float32), out_size=out_size, interpolation=cv2.INTER_NEAREST)
            mask_slice = (mask_slice > 0.5).astype(np.float32)
            fused_np = fused_np * mask_slice + ct_slice * (1.0 - mask_slice)

        fused_slices.append(fused_np)

        if out_png_dir is not None:
            save_png_slice(fused_np, os.path.join(out_png_dir, f"fused_{k:04d}.png"))

        if (k + 1) % 20 == 0 or (k + 1) == z_num:
            print(f"Fused {k + 1}/{z_num} slices")

    fused_arr = np.stack(fused_slices, axis=0)  # [z, 256, 256]
    out_img = make_output_like_ct_256(fused_arr, ct_img, out_size=out_size)

    os.makedirs(os.path.dirname(os.path.abspath(out_nii)), exist_ok=True)
    sitk.WriteImage(out_img, out_nii)
    print(f"Saved fused NIfTI: {out_nii}")

    return out_nii


def parse_args():
    parser = argparse.ArgumentParser(description="Fuse registered CT and MRI volumes using a trained 2D MambaFusion model.")
    parser.add_argument("--ct_nii", required=True, help="Path to CT volume in CT space, e.g., CT_from_DICOM.nii.gz")
    parser.add_argument("--mr_nii", required=True, help="Path to registered MRI volume in CT space, e.g., MR_resampled_to_CT.nii.gz")
    parser.add_argument("--weights", default="model_last/my_cross/fusion_model.pth", help="Path to trained fusion model weights")
    parser.add_argument("--out_nii", required=True, help="Output fused NIfTI path")
    parser.add_argument("--mask_nii", default=None, help="Optional MR_valid_mask_in_CT.nii.gz")
    parser.add_argument("--out_png_dir", default=None, help="Optional directory for fused PNG slices")
    parser.add_argument("--out_size", type=int, default=256, help="In-plane output size. Default: 256")
    parser.add_argument("--ct_lower", type=float, default=-300.0, help="Lower CT window bound")
    parser.add_argument("--ct_upper", type=float, default=1200.0, help="Upper CT window bound")
    parser.add_argument("--gpu", type=int, default=0, help="GPU id. Use -1 for CPU")
    parser.add_argument("--no_keep_ct_outside_mask", action="store_true", help="Do not preserve CT outside valid MRI mask")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    fuse_registered_volumes(
        ct_nii=args.ct_nii,
        mr_nii=args.mr_nii,
        weights=args.weights,
        out_nii=args.out_nii,
        mask_nii=args.mask_nii,
        out_png_dir=args.out_png_dir,
        out_size=args.out_size,
        ct_lower=args.ct_lower,
        ct_upper=args.ct_upper,
        gpu=args.gpu,
        keep_ct_outside_mask=not args.no_keep_ct_outside_mask,
    )
