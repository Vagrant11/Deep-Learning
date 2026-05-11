import argparse
import csv
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models import TinyViTReconstructor


def build_test_loader(batch_size, num_workers, max_test_samples):
    dataset = datasets.CIFAR100(
        root="./data",
        train=False,
        download=True,
        transform=transforms.ToTensor(),
    )
    if max_test_samples is not None:
        dataset = Subset(dataset, list(range(max_test_samples)))

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def load_reconstructor(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    ckpt_args = ckpt["args"]
    model = TinyViTReconstructor(
        patch_size=ckpt_args["patch_size"],
        embed_dim=ckpt_args["embed_dim"],
        encoder_depth=ckpt_args["encoder_depth"],
        decoder_depth=ckpt_args["decoder_depth"],
        num_heads=ckpt_args["num_heads"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def psnr_from_mse(mse):
    if mse <= 0:
        return float("inf")
    return 10.0 * math.log10(1.0 / mse)


@torch.no_grad()
def evaluate_reconstruction(model, loader, device, mask_ratio):
    total_images = 0
    full_mse_sum = 0.0
    objective_mse_sum = 0.0

    for x, _ in loader:
        x = x.to(device, non_blocking=True)
        pred_patches, target_patches, mask, _ = model(x, mask_ratio=mask_ratio)

        recon = model.unpatchify(pred_patches).clamp(0.0, 1.0)
        batch_size = x.size(0)

        full_mse = F.mse_loss(recon, x, reduction="none").flatten(1).mean(dim=1)
        full_mse_sum += full_mse.sum().item()

        patch_mse = F.mse_loss(pred_patches, target_patches, reduction="none").mean(dim=-1)
        if mask is None:
            objective_mse = patch_mse.mean(dim=1)
        else:
            denom = mask.sum(dim=1).clamp(min=1.0)
            objective_mse = (patch_mse * mask).sum(dim=1) / denom
        objective_mse_sum += objective_mse.sum().item()

        total_images += batch_size

    full_mse_mean = full_mse_sum / total_images
    objective_mse_mean = objective_mse_sum / total_images
    return {
        "full_image_mse": full_mse_mean,
        "full_image_psnr": psnr_from_mse(full_mse_mean),
        "objective_patch_mse": objective_mse_mean,
        "objective_patch_psnr": psnr_from_mse(objective_mse_mean),
        "num_images": total_images,
    }


def read_exp1_linear_probe(summary_path):
    if not summary_path.exists():
        return {}

    values = {}
    with summary_path.open(newline="") as f:
        for row in csv.DictReader(f):
            values[row["model"]] = float(row["linear_probe_mean"])
    return values


def save_results(rows, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model",
        "eval_mask_ratio",
        "full_image_mse",
        "full_image_psnr",
        "objective_patch_mse",
        "objective_patch_psnr",
        "linear_probe_mean",
        "num_images",
    ]
    with (output_dir / "exp2_vit_reconstruction_vs_representation.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved ViT Exp. 2 results to {output_dir}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Experiment 2: reconstruction quality vs representation quality.")
    parser.add_argument("--ae-checkpoint", type=Path, default=Path("checkpoints/ae_tinyvit_cifar100.pth"))
    parser.add_argument("--mae-checkpoint", type=Path, default=Path("checkpoints/mae_tinyvit_cifar100.pth"))
    parser.add_argument("--exp1-summary", type=Path, default=Path("results/exp1_vit_cifar100/exp1_vit_summary.csv"))
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--mae-mask-ratio", type=float, default=0.75)
    parser.add_argument("--output-dir", type=Path, default=Path("results/exp2_vit_cifar100"))
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    loader = build_test_loader(args.batch_size, args.num_workers, args.max_test_samples)
    linear_probe = read_exp1_linear_probe(args.exp1_summary)

    configs = [
        ("AE", args.ae_checkpoint, 0.0),
        ("MAE", args.mae_checkpoint, args.mae_mask_ratio),
    ]

    rows = []
    for name, checkpoint, mask_ratio in configs:
        print(f"\n=== ViT Exp. 2 Reconstruction: {name}, mask_ratio={mask_ratio} ===", flush=True)
        model = load_reconstructor(checkpoint, device)
        metrics = evaluate_reconstruction(model, loader, device, mask_ratio)
        row = {
            "model": name,
            "eval_mask_ratio": mask_ratio,
            "linear_probe_mean": linear_probe.get(name, ""),
            **metrics,
        }
        rows.append(row)
        print(
            f"{name}: full_mse={metrics['full_image_mse']:.6f}, "
            f"full_psnr={metrics['full_image_psnr']:.2f}, "
            f"objective_mse={metrics['objective_patch_mse']:.6f}, "
            f"linear_probe={row['linear_probe_mean']}",
            flush=True,
        )

    save_results(rows, args.output_dir)


if __name__ == "__main__":
    main()
