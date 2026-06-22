import os
import sys
import cv2
import time
import random
import argparse
import datetime
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# Make sure this script can import project modules when placed in your project folder
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from vmamba_Fusion_efficross import VSSM_Fusion
from loss import Fusionloss


def set_seed(seed: int = 123):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(path: str):
    if path:
        os.makedirs(path, exist_ok=True)


def read_train_list(train_list: str) -> List[Tuple[str, str]]:
    """
    Read train list.

    Each line supports:
        ct.png,mri.png
    or:
        ct.png mri.png
    """
    samples = []

    with open(train_list, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            if "," in line:
                parts = [p.strip() for p in line.split(",")]
            else:
                parts = line.split()

            if len(parts) != 2:
                raise ValueError(
                    f"Invalid line {line_no} in {train_list}. "
                    f"Expected: ct_image_path,mri_image_path, but got: {line}"
                )

            ct_path, mr_path = parts[0], parts[1]

            if not os.path.exists(ct_path):
                raise FileNotFoundError(f"CT image not found: {ct_path}")

            if not os.path.exists(mr_path):
                raise FileNotFoundError(f"MRI image not found: {mr_path}")

            samples.append((ct_path, mr_path))

    if len(samples) == 0:
        raise RuntimeError(f"No valid training samples found in: {train_list}")

    return samples


def read_gray_image(
    path: str,
    out_size: int = 256,
    normalize_mode: str = "auto"
) -> np.ndarray:
    """
    Read image as grayscale float32 array with shape [1, H, W].

    normalize_mode:
        auto   : automatically normalize to [0, 1]
        255    : divide by 255
        minmax : per-image min-max normalization
        none   : no normalization, only convert to float32

    Supported image formats:
        png, jpg, jpeg, bmp, tif, tiff
    """

    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    if img is None:
        raise FileNotFoundError(f"Failed to read image: {path}")

    # If RGB/BGR image, convert to grayscale
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = img.astype(np.float32)

    # Resize to 256 x 256 if needed
    if img.shape[0] != out_size or img.shape[1] != out_size:
        img = cv2.resize(img, (out_size, out_size), interpolation=cv2.INTER_CUBIC)

    # Normalize
    if normalize_mode == "auto":
        min_v = float(np.min(img))
        max_v = float(np.max(img))

        if min_v >= 0.0 and max_v <= 1.0:
            pass
        elif max_v <= 255.0:
            img = img / 255.0
        else:
            img = (img - min_v) / (max_v - min_v + 1e-8)

    elif normalize_mode == "255":
        img = img / 255.0

    elif normalize_mode == "minmax":
        min_v = float(np.min(img))
        max_v = float(np.max(img))
        img = (img - min_v) / (max_v - min_v + 1e-8)

    elif normalize_mode == "none":
        pass

    else:
        raise ValueError(f"Unsupported normalize_mode: {normalize_mode}")

    img = np.clip(img, 0.0, 1.0).astype(np.float32)

    # [H, W] -> [1, H, W]
    img = img[None, :, :]

    return img


class CTMR2DImageDataset(Dataset):
    def __init__(
        self,
        samples: List[Tuple[str, str]],
        out_size: int = 256,
        normalize_mode: str = "auto"
    ):
        self.samples = samples
        self.out_size = out_size
        self.normalize_mode = normalize_mode

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        ct_path, mr_path = self.samples[index]

        ct = read_gray_image(
            ct_path,
            out_size=self.out_size,
            normalize_mode=self.normalize_mode
        )

        mr = read_gray_image(
            mr_path,
            out_size=self.out_size,
            normalize_mode=self.normalize_mode
        )

        ct = torch.from_numpy(ct).float()
        mr = torch.from_numpy(mr).float()

        name = os.path.basename(ct_path)

        return ct, mr, name


def save_tensor_image(tensor_img: torch.Tensor, save_path: str):
    """
    Save tensor image [1, H, W] or [H, W] to PNG.
    """
    img = tensor_img.detach().cpu().float()

    if img.ndim == 3:
        img = img.squeeze(0)

    img = torch.clamp(img, 0.0, 1.0).numpy()
    img = (img * 255.0).astype(np.uint8)

    cv2.imwrite(save_path, img)


def train_fusion(args):
    # The original Fusionloss has .cuda() in Sobelxy, so CUDA is required.
    if not torch.cuda.is_available() or args.gpu < 0:
        raise RuntimeError(
            "CUDA GPU is required because the original Fusionloss uses .cuda()."
        )

    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(args.gpu)

    print(f"Using device: {device}")

    set_seed(args.seed)
    ensure_dir(args.save_dir)

    samples = read_train_list(args.train_list)

    dataset = CTMR2DImageDataset(
        samples=samples,
        out_size=args.out_size,
        normalize_mode=args.normalize_mode
    )

    train_loader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )

    print(f"Training image pairs: {len(dataset)}")
    print(f"Batch size: {args.batch_size}")
    print(f"Iterations per epoch: {len(train_loader)}")

    model = VSSM_Fusion(in_chans=1).to(device)
    model.train()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    criterion = Fusionloss()

    final_weight_path = os.path.join(args.save_dir, "fusion_model.pth")
    last_weight_path = os.path.join(args.save_dir, "fusion_model_last.pth")

    debug_dir = os.path.join(args.save_dir, "debug_fused")
    ensure_dir(debug_dir)

    best_epoch_loss = float("inf")
    global_step = 0
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
            ct = ct.to(device, non_blocking=True)  # [B, 1, 256, 256]
            mr = mr.to(device, non_blocking=True)  # [B, 1, 256, 256]

            fused = model(ct, mr)
            fused = torch.clamp(fused, 0.0, 1.0)

            loss_total, loss_in, loss_ssim, loss_grad = criterion(
                image_vis=ct,
                image_ir=mr,
                labels=None,
                generate_img=fused,
                i=0
            )

            optimizer.zero_grad(set_to_none=True)
            loss_total.backward()

            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            optimizer.step()

            global_step += 1

            epoch_loss += float(loss_total.item())
            epoch_loss_in += float(loss_in.item())
            epoch_loss_ssim += float(loss_ssim.item())
            epoch_loss_grad += float(loss_grad.item())

            if global_step % args.log_interval == 0:
                elapsed = time.time() - start_time
                done_steps = epoch * len(train_loader) + it + 1
                total_steps = args.epochs * len(train_loader)
                eta_seconds = int(
                    elapsed / max(done_steps, 1) * (total_steps - done_steps)
                )
                eta = str(datetime.timedelta(seconds=eta_seconds))

                print(
                    f"Epoch [{epoch + 1}/{args.epochs}] "
                    f"Iter [{it + 1}/{len(train_loader)}] "
                    f"Step {global_step} | "
                    f"lr={lr_this_epoch:.6g} | "
                    f"loss={loss_total.item():.4f} | "
                    f"loss_in={loss_in.item():.4f} | "
                    f"loss_ssim={loss_ssim.item():.4f} | "
                    f"loss_grad={loss_grad.item():.4f} | "
                    f"eta={eta}"
                )

            # Save several fused images for visual checking
            if args.save_debug and epoch == 0 and it == 0:
                for b in range(min(ct.shape[0], 4)):
                    save_tensor_image(
                        ct[b],
                        os.path.join(debug_dir, f"epoch{epoch+1}_sample{b}_ct.png")
                    )
                    save_tensor_image(
                        mr[b],
                        os.path.join(debug_dir, f"epoch{epoch+1}_sample{b}_mr.png")
                    )
                    save_tensor_image(
                        fused[b],
                        os.path.join(debug_dir, f"epoch{epoch+1}_sample{b}_fused.png")
                    )

        n_iter = max(len(train_loader), 1)

        epoch_loss /= n_iter
        epoch_loss_in /= n_iter
        epoch_loss_ssim /= n_iter
        epoch_loss_grad /= n_iter

        print(
            f"\nEpoch {epoch + 1}/{args.epochs} finished | "
            f"avg_loss={epoch_loss:.4f}, "
            f"avg_loss_in={epoch_loss_in:.4f}, "
            f"avg_loss_ssim={epoch_loss_ssim:.4f}, "
            f"avg_loss_grad={epoch_loss_grad:.4f}\n"
        )

        # Save last checkpoint every epoch
        torch.save(model.state_dict(), last_weight_path)

        # Save best checkpoint according to training loss
        if epoch_loss < best_epoch_loss:
            best_epoch_loss = epoch_loss
            torch.save(model.state_dict(), final_weight_path)
            print(f"Saved best model to: {final_weight_path}")

        if args.save_epoch_checkpoints:
            epoch_weight_path = os.path.join(
                args.save_dir,
                f"fusion_model_epoch_{epoch + 1:03d}.pth"
            )
            torch.save(model.state_dict(), epoch_weight_path)

    if not os.path.exists(final_weight_path):
        torch.save(model.state_dict(), final_weight_path)

    print("Training finished.")
    print(f"Best training loss: {best_epoch_loss:.4f}")
    print(f"Best model path: {final_weight_path}")
    print(f"Last model path: {last_weight_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train 2D MambaFusion using paired CT-MRI images."
    )

    parser.add_argument(
        "--train_list",
        required=True,
        help="TXT file. Each line: ct_image_path,mri_image_path"
    )

    parser.add_argument(
        "--save_dir",
        default=os.path.join("model_last", "my_cross"),
        help="Directory to save fusion_model.pth"
    )

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr_decay", type=float, default=0.75)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--grad_clip", type=float, default=0.0)

    parser.add_argument(
        "--out_size",
        type=int,
        default=256,
        help="Input image size. Default: 256"
    )

    parser.add_argument(
        "--normalize_mode",
        default="auto",
        choices=["auto", "255", "minmax", "none"],
        help="Image normalization mode"
    )

    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--log_interval", type=int, default=10)

    parser.add_argument(
        "--save_debug",
        action="store_true",
        help="Save several CT/MRI/fused images for visual checking"
    )

    parser.add_argument(
        "--save_epoch_checkpoints",
        action="store_true",
        help="Save checkpoint for each epoch"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_fusion(args)