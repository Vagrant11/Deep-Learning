import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models import FeatureEncoder, TinyViTReconstructor
from src.probing.linear_probe import LinearProbe, evaluate, train_epoch
from src.shape_texture import FourierConflictDataset


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def mean_std(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 1:
        return float(values.mean()), 0.0
    return float(values.mean()), float(values.std(ddof=1))


def build_loaders(batch_size, num_workers, max_train_samples, max_test_samples, seed, alpha, offset):
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )
    test_transform = transforms.ToTensor()

    train_dataset = datasets.CIFAR100(root="./data", train=True, download=True, transform=train_transform)
    clean_test_dataset = datasets.CIFAR100(root="./data", train=False, download=True, transform=test_transform)

    if max_train_samples is not None:
        train_dataset = Subset(train_dataset, list(range(max_train_samples)))
    if max_test_samples is not None:
        clean_test_dataset = Subset(clean_test_dataset, list(range(max_test_samples)))

    conflict_dataset = FourierConflictDataset(clean_test_dataset, alpha=alpha, offset=offset)

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
    clean_test_loader = DataLoader(
        clean_test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    conflict_loader = DataLoader(
        conflict_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, clean_test_loader, conflict_loader


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


@torch.no_grad()
def evaluate_conflict(probe, loader, device):
    probe.eval()
    total = 0
    shape_wins = 0
    texture_wins = 0
    other_predictions = 0
    shape_correct = 0
    texture_correct = 0
    margin_sum = 0.0
    entropy_sum = 0.0

    for x, shape_y, texture_y in loader:
        x = x.to(device, non_blocking=True)
        shape_y = shape_y.to(device, non_blocking=True)
        texture_y = texture_y.to(device, non_blocking=True)

        logits = probe(x)
        predictions = logits.argmax(dim=1)
        shape_logits = logits.gather(1, shape_y.view(-1, 1)).squeeze(1)
        texture_logits = logits.gather(1, texture_y.view(-1, 1)).squeeze(1)
        probabilities = F.softmax(logits, dim=1)
        entropy = -(probabilities * probabilities.clamp(min=1e-12).log()).sum(dim=1)

        batch_size = x.size(0)
        total += batch_size
        shape_wins += (shape_logits > texture_logits).sum().item()
        texture_wins += (texture_logits > shape_logits).sum().item()
        shape_correct += (predictions == shape_y).sum().item()
        texture_correct += (predictions == texture_y).sum().item()
        other_predictions += ((predictions != shape_y) & (predictions != texture_y)).sum().item()
        margin_sum += (shape_logits - texture_logits).sum().item()
        entropy_sum += entropy.sum().item()

    return {
        "shape_bias_rate": shape_wins / total,
        "texture_bias_rate": texture_wins / total,
        "shape_top1_rate": shape_correct / total,
        "texture_top1_rate": texture_correct / total,
        "other_top1_rate": other_predictions / total,
        "shape_minus_texture_logit": margin_sum / total,
        "prediction_entropy": entropy_sum / total,
        "num_conflict_images": total,
    }


def run_probe(name, encoder, train_loader, test_loader, conflict_loader, device, epochs, seed, feature_dim):
    print(f"\n=== ViT Exp. 4 Shape/Texture: {name}, seed={seed} ===", flush=True)
    probe = LinearProbe(encoder, feature_dim=feature_dim, num_classes=100).to(device)
    optimizer = torch.optim.AdamW(probe.classifier.parameters(), lr=1e-3, weight_decay=0.0)

    history = []
    for epoch in range(epochs):
        loss, train_acc = train_epoch(probe, train_loader, optimizer, device)
        clean_acc = evaluate(probe, test_loader, device)
        history.append(
            {
                "model": name,
                "seed": seed,
                "epoch": epoch,
                "loss": loss,
                "train_acc": train_acc,
                "clean_acc": clean_acc,
            }
        )
        print(f"Epoch {epoch}: loss={loss:.4f} train={train_acc:.4f} clean={clean_acc:.4f}", flush=True)

    conflict_metrics = evaluate_conflict(probe, conflict_loader, device)
    conflict_metrics["clean_acc"] = history[-1]["clean_acc"]
    return conflict_metrics, history


def save_results(summary_rows, seed_rows, histories, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "model",
        "clean_acc_mean",
        "clean_acc_std",
        "shape_bias_rate_mean",
        "shape_bias_rate_std",
        "texture_bias_rate_mean",
        "texture_bias_rate_std",
        "shape_top1_rate_mean",
        "texture_top1_rate_mean",
        "other_top1_rate_mean",
        "shape_minus_texture_logit_mean",
        "prediction_entropy_mean",
        "seeds",
        "num_conflict_images",
    ]
    with (output_dir / "exp4_shape_texture_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    seed_fieldnames = [
        "model",
        "seed",
        "clean_acc",
        "shape_bias_rate",
        "texture_bias_rate",
        "shape_top1_rate",
        "texture_top1_rate",
        "other_top1_rate",
        "shape_minus_texture_logit",
        "prediction_entropy",
        "num_conflict_images",
    ]
    with (output_dir / "exp4_shape_texture_by_seed.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=seed_fieldnames)
        writer.writeheader()
        writer.writerows(seed_rows)

    with (output_dir / "exp4_shape_texture_history.json").open("w") as f:
        json.dump(histories, f, indent=2)

    print(f"\nSaved ViT Exp. 4 results to {output_dir}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Experiment 4: Fourier shape-texture conflict for tiny ViT CIFAR-100.")
    parser.add_argument("--ae-checkpoint", type=Path, default=Path("checkpoints/ae_tinyvit_cifar100.pth"))
    parser.add_argument("--mae-checkpoint", type=Path, default=Path("checkpoints/mae_tinyvit_cifar100.pth"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--texture-offset", type=int, default=137)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=5000)
    parser.add_argument("--output-dir", type=Path, default=Path("results/exp4_shape_texture"))
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    ae_model = load_reconstructor(args.ae_checkpoint, device)
    mae_model = load_reconstructor(args.mae_checkpoint, device)
    feature_dim = ae_model.embed_dim
    encoders = {
        "AE": FeatureEncoder(ae_model).to(device),
        "MAE": FeatureEncoder(mae_model).to(device),
    }

    seed_rows = []
    histories = {}
    for name, encoder in encoders.items():
        histories[name] = {}
        for seed in args.seeds:
            set_seed(seed)
            train_loader, test_loader, conflict_loader = build_loaders(
                args.batch_size,
                args.num_workers,
                args.max_train_samples,
                args.max_test_samples,
                seed,
                args.alpha,
                args.texture_offset,
            )
            metrics, history = run_probe(
                name, encoder, train_loader, test_loader, conflict_loader, device, args.epochs, seed, feature_dim
            )
            seed_rows.append({"model": name, "seed": seed, **metrics})
            histories[name][str(seed)] = history

    summary_rows = []
    for name in encoders:
        model_rows = [row for row in seed_rows if row["model"] == name]
        clean_mean, clean_std = mean_std([row["clean_acc"] for row in model_rows])
        shape_mean, shape_std = mean_std([row["shape_bias_rate"] for row in model_rows])
        texture_mean, texture_std = mean_std([row["texture_bias_rate"] for row in model_rows])
        summary_rows.append(
            {
                "model": name,
                "clean_acc_mean": clean_mean,
                "clean_acc_std": clean_std,
                "shape_bias_rate_mean": shape_mean,
                "shape_bias_rate_std": shape_std,
                "texture_bias_rate_mean": texture_mean,
                "texture_bias_rate_std": texture_std,
                "shape_top1_rate_mean": np.mean([row["shape_top1_rate"] for row in model_rows]),
                "texture_top1_rate_mean": np.mean([row["texture_top1_rate"] for row in model_rows]),
                "other_top1_rate_mean": np.mean([row["other_top1_rate"] for row in model_rows]),
                "shape_minus_texture_logit_mean": np.mean(
                    [row["shape_minus_texture_logit"] for row in model_rows]
                ),
                "prediction_entropy_mean": np.mean([row["prediction_entropy"] for row in model_rows]),
                "seeds": len(model_rows),
                "num_conflict_images": model_rows[0]["num_conflict_images"],
            }
        )

    save_results(summary_rows, seed_rows, histories, args.output_dir)


if __name__ == "__main__":
    main()
