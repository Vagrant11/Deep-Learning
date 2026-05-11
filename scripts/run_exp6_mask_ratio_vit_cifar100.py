import argparse
import csv
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models import FeatureEncoder, TinyViTReconstructor, reconstruction_loss
from src.probing.linear_probe import LinearProbe, evaluate, train_epoch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ratio_tag(mask_ratio):
    return f"{int(round(mask_ratio * 100)):03d}"


def build_pretrain_loader(batch_size, num_workers, max_train_samples, seed):
    transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )
    dataset = datasets.CIFAR100(root="./data", train=True, download=True, transform=transform)
    if max_train_samples is not None:
        dataset = Subset(dataset, list(range(max_train_samples)))

    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )


def build_probe_loaders(batch_size, num_workers, max_train_samples, max_test_samples, seed):
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )
    test_transform = transforms.ToTensor()
    train_dataset = datasets.CIFAR100(root="./data", train=True, download=True, transform=train_transform)
    test_dataset = datasets.CIFAR100(root="./data", train=False, download=True, transform=test_transform)

    if max_train_samples is not None:
        train_dataset = Subset(train_dataset, list(range(max_train_samples)))
    if max_test_samples is not None:
        test_dataset = Subset(test_dataset, list(range(max_test_samples)))

    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, test_loader


def build_model(args):
    return TinyViTReconstructor(
        image_size=32,
        patch_size=args.patch_size,
        embed_dim=args.embed_dim,
        encoder_depth=args.encoder_depth,
        decoder_depth=args.decoder_depth,
        num_heads=args.num_heads,
    )


def save_checkpoint(model, optimizer, args, mask_ratio, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_type": "AE" if mask_ratio == 0 else "MAE",
            "mask_ratio": mask_ratio,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "args": vars(args),
        },
        path,
    )


def load_checkpoint(path, args, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    ckpt_args = ckpt.get("args", {})
    model = TinyViTReconstructor(
        patch_size=ckpt_args.get("patch_size", args.patch_size),
        embed_dim=ckpt_args.get("embed_dim", args.embed_dim),
        encoder_depth=ckpt_args.get("encoder_depth", args.encoder_depth),
        decoder_depth=ckpt_args.get("decoder_depth", args.decoder_depth),
        num_heads=ckpt_args.get("num_heads", args.num_heads),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    return model


def train_one_epoch(model, loader, optimizer, scaler, device, mask_ratio, use_amp):
    model.train()
    total_loss = 0.0
    for x, _ in loader:
        x = x.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            pred_patches, target_patches, mask, _ = model(x, mask_ratio=mask_ratio)
            loss = reconstruction_loss(pred_patches, target_patches, mask)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
    return total_loss / len(loader)


def pretrain_ratio(mask_ratio, args, loader, device):
    checkpoint_path = args.checkpoint_dir / f"tinyvit_mask_ratio_{ratio_tag(mask_ratio)}.pth"
    if checkpoint_path.exists() and not args.force_pretrain:
        print(f"\n=== Reusing checkpoint for mask_ratio={mask_ratio}: {checkpoint_path} ===", flush=True)
        return load_checkpoint(checkpoint_path, args, device), checkpoint_path, []

    model = build_model(args).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    use_amp = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    print(f"\n=== Exp. 6 pretrain mask_ratio={mask_ratio} ===", flush=True)
    losses = []
    for epoch in range(args.pretrain_epochs):
        loss = train_one_epoch(model, loader, optimizer, scaler, device, mask_ratio, use_amp)
        losses.append(loss)
        print(f"mask_ratio={mask_ratio} epoch {epoch}: loss={loss:.5f}", flush=True)

    save_checkpoint(model, optimizer, args, mask_ratio, checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}", flush=True)
    return model, checkpoint_path, losses


def run_linear_probe(mask_ratio, model, args, device, seed):
    train_loader, test_loader = build_probe_loaders(
        args.batch_size,
        args.num_workers,
        args.max_probe_train_samples,
        args.max_probe_test_samples,
        seed,
    )
    encoder = FeatureEncoder(model).to(device)
    probe = LinearProbe(encoder, feature_dim=model.embed_dim, num_classes=100).to(device)
    optimizer = torch.optim.AdamW(probe.classifier.parameters(), lr=args.probe_learning_rate, weight_decay=0.0)

    print(f"\n=== Exp. 6 linear probe mask_ratio={mask_ratio}, seed={seed} ===", flush=True)
    test_acc = 0.0
    for epoch in range(args.probe_epochs):
        loss, train_acc = train_epoch(probe, train_loader, optimizer, device)
        test_acc = evaluate(probe, test_loader, device)
        print(f"epoch {epoch}: loss={loss:.4f} train={train_acc:.4f} test={test_acc:.4f}", flush=True)
    return test_acc


def mean_std(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 1:
        return float(values.mean()), 0.0
    return float(values.mean()), float(values.std(ddof=1))


def save_results(summary_rows, seed_rows, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "exp6_mask_ratio_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "mask_ratio",
                "linear_probe_mean",
                "linear_probe_std",
                "linear_probe_seeds",
                "checkpoint",
                "final_pretrain_loss",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    with (output_dir / "exp6_mask_ratio_by_seed.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["mask_ratio", "seed", "linear_probe_acc"])
        writer.writeheader()
        writer.writerows(seed_rows)

    print(f"\nSaved Exp. 6 results to {output_dir}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Experiment 6: mask-ratio ablation with tiny ViT on CIFAR-100.")
    parser.add_argument("--mask-ratios", type=float, nargs="+", default=[0.0, 0.15, 0.3, 0.5, 0.75, 0.9])
    parser.add_argument("--pretrain-epochs", type=int, default=20)
    parser.add_argument("--probe-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-pretrain-samples", type=int, default=None)
    parser.add_argument("--max-probe-train-samples", type=int, default=None)
    parser.add_argument("--max-probe-test-samples", type=int, default=None)

    parser.add_argument("--patch-size", type=int, default=4)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--encoder-depth", type=int, default=4)
    parser.add_argument("--decoder-depth", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--probe-learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--force-pretrain", action="store_true")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints/exp6_mask_ratio_vit_cifar100"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/exp6_mask_ratio_vit_cifar100"))
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    pretrain_loader = build_pretrain_loader(
        args.batch_size,
        args.num_workers,
        args.max_pretrain_samples,
        args.seed,
    )

    summary_rows = []
    seed_rows = []
    for mask_ratio in args.mask_ratios:
        model, checkpoint_path, losses = pretrain_ratio(mask_ratio, args, pretrain_loader, device)
        model.eval()

        accs = []
        for seed in args.seeds:
            set_seed(seed)
            acc = run_linear_probe(mask_ratio, model, args, device, seed)
            accs.append(acc)
            seed_rows.append({"mask_ratio": mask_ratio, "seed": seed, "linear_probe_acc": acc})

        mean, std = mean_std(accs)
        summary_rows.append(
            {
                "mask_ratio": mask_ratio,
                "linear_probe_mean": mean,
                "linear_probe_std": std,
                "linear_probe_seeds": len(args.seeds),
                "checkpoint": str(checkpoint_path),
                "final_pretrain_loss": losses[-1] if losses else "",
            }
        )
        save_results(summary_rows, seed_rows, args.output_dir)


if __name__ == "__main__":
    main()
