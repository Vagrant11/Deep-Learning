import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models import FeatureEncoder, TinyViTReconstructor
from src.probing.linear_probe import LinearProbe, evaluate, train_epoch
from src.robustness import PerturbedDataset


PERTURBATIONS = ["clean", "high_pass", "patch_shuffle", "patch_drop"]


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


def build_train_loader(batch_size, num_workers, max_train_samples, seed):
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )
    dataset = datasets.CIFAR100(root="./data", train=True, download=True, transform=train_transform)
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


def build_eval_loaders(
    batch_size,
    num_workers,
    max_test_samples,
    seed,
    patch_size,
    drop_ratio,
    high_pass_kernel,
    high_pass_strength,
):
    dataset = datasets.CIFAR100(root="./data", train=False, download=True, transform=transforms.ToTensor())
    if max_test_samples is not None:
        dataset = Subset(dataset, list(range(max_test_samples)))

    loaders = {}
    for perturbation in PERTURBATIONS:
        perturbed = PerturbedDataset(
            dataset,
            perturbation=perturbation,
            patch_size=patch_size,
            drop_ratio=drop_ratio,
            high_pass_kernel=high_pass_kernel,
            high_pass_strength=high_pass_strength,
            seed=seed,
        )
        loaders[perturbation] = DataLoader(
            perturbed,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
    return loaders


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


def run_probe(name, encoder, train_loader, eval_loaders, device, epochs, seed, feature_dim):
    print(f"\n=== ViT Exp. 5 Robustness: {name}, seed={seed} ===", flush=True)
    probe = LinearProbe(encoder, feature_dim=feature_dim, num_classes=100).to(device)
    optimizer = torch.optim.AdamW(probe.classifier.parameters(), lr=1e-3, weight_decay=0.0)

    history = []
    for epoch in range(epochs):
        loss, train_acc = train_epoch(probe, train_loader, optimizer, device)
        clean_acc = evaluate(probe, eval_loaders["clean"], device)
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

    rows = []
    clean_acc = None
    for perturbation, loader in eval_loaders.items():
        accuracy = evaluate(probe, loader, device)
        if perturbation == "clean":
            clean_acc = accuracy
        rows.append({"perturbation": perturbation, "accuracy": accuracy})
        print(f"{name} seed={seed} {perturbation}: acc={accuracy:.4f}", flush=True)

    for row in rows:
        row["absolute_drop"] = clean_acc - row["accuracy"]
        row["relative_drop"] = (clean_acc - row["accuracy"]) / max(clean_acc, 1e-12)

    return rows, history


def save_results(summary_rows, seed_rows, histories, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_fields = [
        "model",
        "perturbation",
        "accuracy_mean",
        "accuracy_std",
        "absolute_drop_mean",
        "absolute_drop_std",
        "relative_drop_mean",
        "relative_drop_std",
        "seeds",
    ]
    with (output_dir / "exp5_robustness_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    seed_fields = ["model", "seed", "perturbation", "accuracy", "absolute_drop", "relative_drop"]
    with (output_dir / "exp5_robustness_by_seed.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=seed_fields)
        writer.writeheader()
        writer.writerows(seed_rows)

    with (output_dir / "exp5_robustness_history.json").open("w") as f:
        json.dump(histories, f, indent=2)

    print(f"\nSaved ViT Exp. 5 results to {output_dir}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Experiment 5: perturbation robustness for tiny ViT CIFAR-100.")
    parser.add_argument("--ae-checkpoint", type=Path, default=Path("checkpoints/ae_tinyvit_cifar100.pth"))
    parser.add_argument("--mae-checkpoint", type=Path, default=Path("checkpoints/mae_tinyvit_cifar100.pth"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--patch-size", type=int, default=4)
    parser.add_argument("--drop-ratio", type=float, default=0.5)
    parser.add_argument("--high-pass-kernel", type=int, default=5)
    parser.add_argument("--high-pass-strength", type=float, default=1.0)
    parser.add_argument("--output-dir", type=Path, default=Path("results/exp5_robustness"))
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
            train_loader = build_train_loader(args.batch_size, args.num_workers, args.max_train_samples, seed)
            eval_loaders = build_eval_loaders(
                args.batch_size,
                args.num_workers,
                args.max_test_samples,
                seed,
                args.patch_size,
                args.drop_ratio,
                args.high_pass_kernel,
                args.high_pass_strength,
            )
            rows, history = run_probe(name, encoder, train_loader, eval_loaders, device, args.epochs, seed, feature_dim)
            histories[name][str(seed)] = history
            for row in rows:
                seed_rows.append({"model": name, "seed": seed, **row})

    summary_rows = []
    for name in encoders:
        for perturbation in PERTURBATIONS:
            rows = [row for row in seed_rows if row["model"] == name and row["perturbation"] == perturbation]
            acc_mean, acc_std = mean_std([row["accuracy"] for row in rows])
            abs_mean, abs_std = mean_std([row["absolute_drop"] for row in rows])
            rel_mean, rel_std = mean_std([row["relative_drop"] for row in rows])
            summary_rows.append(
                {
                    "model": name,
                    "perturbation": perturbation,
                    "accuracy_mean": acc_mean,
                    "accuracy_std": acc_std,
                    "absolute_drop_mean": abs_mean,
                    "absolute_drop_std": abs_std,
                    "relative_drop_mean": rel_mean,
                    "relative_drop_std": rel_std,
                    "seeds": len(rows),
                }
            )

    save_results(summary_rows, seed_rows, histories, args.output_dir)


if __name__ == "__main__":
    main()
