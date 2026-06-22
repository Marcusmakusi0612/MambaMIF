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
    """Load trained MambaFusion model."""
    model = VSSM_Fusion(in_chans=1).to(device)

    state = torch.load(weights_path, map_location=device)

    if isinstance(state, dict):
        if "state_dict" in state:
            state = state["state_dict"]
        elif "model" in state:
            state = state["model"]

    state = strip_module_prefix(state)

    model.load_state_dict(state, strict=True)
    model.eval()

    return model


def sitk_to_array(image):
    """SimpleITK image -> numpy array with shape [z, y, x]."""
    return sitk.GetArrayFromImage(image).astype(np.float32)


def normalize_ct(ct_arr, lower=-300.0, upper=1200.0):
    """
    CT windowing + normalization to [0, 1].
    """
    ct_arr = ct_arr.astype(np.float32)
    ct_arr = np.clip(ct_arr, lower, upper)
    ct_arr = (ct_arr - lower) / (upper - lower)
    ct_arr = np.clip(ct_arr, 0.0, 1.0).astype(np.float32)
    return ct_arr


def normalize_mri(mr_arr, mask_arr=None, eps=1e-8):
    """
    MRI normalization:
      z-score normalization followed by rescaling to [0, 1].

    If a valid MRI mask is provided, statistics are computed inside the mask.
    """
    mr_arr = mr_arr.astype(np.float32)

    if mask_arr is not None:
        valid = mask_arr > 0
    else:
        valid = np.isfinite(mr_arr) & (mr_arr != 0)

        # If the MRI is not zero-background or valid region is too small,
        # use all finite voxels instead.
        if valid.sum() < 10:
            valid = np.isfinite(mr_arr)

    values = mr_arr[valid]

    if values.size == 0:
        print("Warning: no valid MRI voxels found. Return zero MRI volume.")
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


def resize_slice(slice_2d, out_size=256, interpolation=cv2.INTER_CUBIC):
    """Resize a 2D slice to out_size x out_size."""
    return cv2.resize(
        slice_2d.astype(np.float32),
        (out_size, out_size),
        interpolation=interpolation
    )


def save_png_slice(image_2d_float, output_path):
    """Save a normalized float image [0, 1] as PNG."""
    image_uint8 = np.clip(image_2d_float * 255.0, 0, 255).astype(np.uint8)
    cv2.imwrite(output_path, image_uint8)


def make_output_like_ct_256(fused_arr_zyx, ct_img, out_size=256):
    """
    Create SimpleITK image from fused array [z, 256, 256].

    The in-plane spacing is adjusted so that the output image preserves
    the physical field of view of the original CT volume.
    """
    out_img = sitk.GetImageFromArray(fused_arr_zyx.astype(np.float32))

    ct_size = ct_img.GetSize()        # (x, y, z)
    ct_spacing = ct_img.GetSpacing()  # (sx, sy, sz)

    new_spacing = (
        ct_spacing[0] * ct_size[0] / float(out_size),
        ct_spacing[1] * ct_size[1] / float(out_size),
        ct_spacing[2],
    )

    out_img.SetSpacing(new_spacing)
    out_img.SetOrigin(ct_img.GetOrigin())
    out_img.SetDirection(ct_img.GetDirection())

    return out_img


def make_output_like_ct_original(fused_arr_zyx, ct_img):
    """
    Create SimpleITK image from fused array [z, y, x].

    The output image has the same size, spacing, origin, and direction as CT.
    This is recommended for spatial alignment and visualization.
    """
    out_img = sitk.GetImageFromArray(fused_arr_zyx.astype(np.float32))
    out_img.CopyInformation(ct_img)
    return out_img


def fuse_registered_volumes(
    ct_nii,
    mr_nii,
    weights,
    out_nii,
    mask_nii=None,
    out_png_dir=None,
    out_size=256,
    output_mode="ct_size",
    ct_lower=-300.0,
    ct_upper=1200.0,
    gpu=0,
    keep_ct_outside_mask=True,
):
    """
    Fuse registered CT-MRI 3D volumes slice by slice.

    output_mode:
      "256"     : output fused NIfTI size is [z, 256, 256]
      "ct_size" : output fused NIfTI has the same size as CT
    """

    if output_mode not in ["256", "ct_size"]:
        raise ValueError("output_mode must be '256' or 'ct_size'.")

    device = torch.device(
        f"cuda:{gpu}" if torch.cuda.is_available() and gpu >= 0 else "cpu"
    )
    print(f"Using device: {device}")

    print(f"Reading CT : {ct_nii}")
    print(f"Reading MRI: {mr_nii}")

    ct_img = sitk.ReadImage(ct_nii)
    mr_img = sitk.ReadImage(mr_nii)

    if ct_img.GetSize() != mr_img.GetSize():
        raise ValueError(
            f"CT and MRI sizes are different: CT={ct_img.GetSize()}, MRI={mr_img.GetSize()}.\n"
            f"Please use MR_resampled_to_CT.nii.gz or resample MRI into CT space first."
        )

    if ct_img.GetSpacing() != mr_img.GetSpacing():
        print(
            f"Warning: CT and MRI spacing are different: "
            f"CT={ct_img.GetSpacing()}, MRI={mr_img.GetSpacing()}."
        )

    if ct_img.GetOrigin() != mr_img.GetOrigin():
        print(
            f"Warning: CT and MRI origin are different: "
            f"CT={ct_img.GetOrigin()}, MRI={mr_img.GetOrigin()}."
        )

    if ct_img.GetDirection() != mr_img.GetDirection():
        print("Warning: CT and MRI direction matrices are different.")

    ct_arr_raw = sitk_to_array(ct_img)  # [z, y, x]
    mr_arr_raw = sitk_to_array(mr_img)  # [z, y, x]

    z_num, h_ori, w_ori = ct_arr_raw.shape
    print(f"Input volume array shape [z, y, x]: {ct_arr_raw.shape}")

    mask_arr = None
    if mask_nii is not None and os.path.exists(mask_nii):
        print(f"Reading mask: {mask_nii}")
        mask_img = sitk.ReadImage(mask_nii)

        if mask_img.GetSize() != ct_img.GetSize():
            raise ValueError(
                f"Mask size {mask_img.GetSize()} does not match CT size {ct_img.GetSize()}."
            )

        mask_arr = sitk_to_array(mask_img) > 0

    ct_arr = normalize_ct(ct_arr_raw, lower=ct_lower, upper=ct_upper)
    mr_arr = normalize_mri(mr_arr_raw, mask_arr=mask_arr)

    print(f"Loading model weights: {weights}")
    model = load_model(weights, device)

    fused_slices = []

    if out_png_dir is not None:
        os.makedirs(out_png_dir, exist_ok=True)

    with torch.inference_mode():
        for k in range(z_num):
            # Original-size slices
            ct_slice_ori = ct_arr[k]  # [h, w]
            mr_slice_ori = mr_arr[k]

            # Resize to model input size 256 x 256
            ct_slice_256 = resize_slice(
                ct_slice_ori,
                out_size=out_size,
                interpolation=cv2.INTER_CUBIC
            )

            mr_slice_256 = resize_slice(
                mr_slice_ori,
                out_size=out_size,
                interpolation=cv2.INTER_CUBIC
            )

            ct_tensor = torch.from_numpy(
                ct_slice_256[None, None, :, :]
            ).float().to(device)

            mr_tensor = torch.from_numpy(
                mr_slice_256[None, None, :, :]
            ).float().to(device)

            fused = model(ct_tensor, mr_tensor)
            fused = torch.clamp(fused, 0.0, 1.0)

            fused_256 = fused.squeeze().detach().cpu().numpy().astype(np.float32)

            # Output mode 1: save 256 x 256 x n
            if output_mode == "256":
                fused_out = fused_256

                if mask_arr is not None and keep_ct_outside_mask:
                    mask_slice_256 = resize_slice(
                        mask_arr[k].astype(np.float32),
                        out_size=out_size,
                        interpolation=cv2.INTER_NEAREST
                    )
                    mask_slice_256 = (mask_slice_256 > 0.5).astype(np.float32)
                    fused_out = fused_out * mask_slice_256 + ct_slice_256 * (1.0 - mask_slice_256)

            # Output mode 2: resize back to original CT size
            else:
                fused_out = cv2.resize(
                    fused_256,
                    (w_ori, h_ori),
                    interpolation=cv2.INTER_CUBIC
                ).astype(np.float32)

                if mask_arr is not None and keep_ct_outside_mask:
                    mask_slice_ori = mask_arr[k].astype(np.float32)
                    mask_slice_ori = (mask_slice_ori > 0.5).astype(np.float32)
                    fused_out = fused_out * mask_slice_ori + ct_slice_ori * (1.0 - mask_slice_ori)

            fused_out = np.clip(fused_out, 0.0, 1.0).astype(np.float32)
            fused_slices.append(fused_out)

            if out_png_dir is not None:
                save_png_slice(
                    fused_out,
                    os.path.join(out_png_dir, f"fused_{k:04d}.png")
                )

            if (k + 1) % 20 == 0 or (k + 1) == z_num:
                print(f"Fused {k + 1}/{z_num} slices")

    fused_arr = np.stack(fused_slices, axis=0).astype(np.float32)

    print(f"Output fused array shape [z, y, x]: {fused_arr.shape}")

    if output_mode == "256":
        out_img = make_output_like_ct_256(
            fused_arr,
            ct_img,
            out_size=out_size
        )
    else:
        out_img = make_output_like_ct_original(
            fused_arr,
            ct_img
        )

    out_dir = os.path.dirname(os.path.abspath(out_nii))
    os.makedirs(out_dir, exist_ok=True)

    sitk.WriteImage(out_img, out_nii)

    print(f"Saved fused NIfTI: {out_nii}")

    return out_nii


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fuse registered CT and MRI volumes using a trained 2D MambaFusion model."
    )

    parser.add_argument(
        "--ct_nii",
        required=True,
        help="Path to CT volume in CT space, e.g. CT_from_DICOM.nii.gz"
    )

    parser.add_argument(
        "--mr_nii",
        required=True,
        help="Path to registered MRI volume in CT space, e.g. MR_resampled_to_CT.nii.gz"
    )

    parser.add_argument(
        "--weights",
        default="model_last/my_cross/fusion_model.pth",
        help="Path to trained fusion model weights"
    )

    parser.add_argument(
        "--out_nii",
        required=True,
        help="Output fused NIfTI path"
    )

    parser.add_argument(
        "--mask_nii",
        default=None,
        help="Optional valid MRI mask in CT space, e.g. MR_valid_mask_in_CT.nii.gz"
    )

    parser.add_argument(
        "--out_png_dir",
        default=None,
        help="Optional directory for saving fused PNG slices"
    )

    parser.add_argument(
        "--out_size",
        type=int,
        default=256,
        help="Model input size. Default: 256"
    )

    parser.add_argument(
        "--output_mode",
        default="ct_size",
        choices=["256", "ct_size"],
        help="Output NIfTI size mode. 'ct_size' is recommended for alignment with CT."
    )

    parser.add_argument(
        "--ct_lower",
        type=float,
        default=-300.0,
        help="Lower CT window bound"
    )

    parser.add_argument(
        "--ct_upper",
        type=float,
        default=1200.0,
        help="Upper CT window bound"
    )

    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU id. Use -1 for CPU"
    )

    parser.add_argument(
        "--no_keep_ct_outside_mask",
        action="store_true",
        help="Do not preserve CT outside valid MRI mask"
    )

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
        output_mode=args.output_mode,
        ct_lower=args.ct_lower,
        ct_upper=args.ct_upper,
        gpu=args.gpu,
        keep_ct_outside_mask=not args.no_keep_ct_outside_mask,
    )