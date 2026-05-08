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
from src.probing.knn import extract_features, knn_accuracy
from src.probing.linear_probe import LinearProbe, evaluate, train_epoch


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


def build_loaders(batch_size, num_workers, max_train_samples, max_test_samples, seed):
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )
    test_transform = transforms.Compose([transforms.ToTensor()])

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


def load_reconstructor(path, device):
    ckpt = torch.load(path, map_location=device)
    ckpt_args = ckpt["args"]
    model = TinyViTReconstructor(
        patch_size=ckpt_args["patch_size"],
        embed_dim=ckpt_args["embed_dim"],
        encoder_depth=ckpt_args["encoder_depth"],
        decoder_depth=ckpt_args["decoder_depth"],
        num_heads=ckpt_args["num_heads"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    return model


def run_linear_probe(name, encoder, train_loader, test_loader, device, epochs, seed, feature_dim):
    print(f"\n=== ViT Exp. 1 Linear Probe: {name}, seed={seed} ===", flush=True)
    probe = LinearProbe(encoder, feature_dim=feature_dim, num_classes=100).to(device)
    optimizer = torch.optim.AdamW(probe.classifier.parameters(), lr=1e-3, weight_decay=0.0)

    history = []
    test_acc = 0.0
    for epoch in range(epochs):
        loss, train_acc = train_epoch(probe, train_loader, optimizer, device)
        test_acc = evaluate(probe, test_loader, device)
        history.append(
            {
                "model": name,
                "seed": seed,
                "epoch": epoch,
                "loss": loss,
                "train_acc": train_acc,
                "test_acc": test_acc,
            }
        )
        print(f"Epoch {epoch}: loss={loss:.4f}  train={train_acc:.4f}  test={test_acc:.4f}", flush=True)

    return test_acc, history


def run_knn(name, encoder, train_loader, test_loader, device, k_values, train_limit, test_limit):
    print(f"\n=== ViT Exp. 1 k-NN: {name} ===", flush=True)
    print(f"Extracting train features, limit={train_limit or 'all'}", flush=True)
    train_features, train_labels = extract_features(encoder, train_loader, device, train_limit)
    print(f"Extracting test features, limit={test_limit or 'all'}", flush=True)
    test_features, test_labels = extract_features(encoder, test_loader, device, test_limit)

    results = []
    for k in k_values:
        acc = knn_accuracy(train_features, train_labels, test_features, test_labels, k=k)
        print(f"{name} k-NN accuracy, k={k}: {acc:.4f}", flush=True)
        results.append({"model": name, "k": k, "knn_acc": acc})
    return results


def save_results(summary_rows, seed_rows, knn_rows, histories, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "exp1_vit_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "linear_probe_mean",
                "linear_probe_std",
                "linear_probe_seeds",
                "best_knn_k",
                "best_knn_acc",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    with (output_dir / "exp1_vit_linear_probe_by_seed.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "seed", "linear_probe_acc"])
        writer.writeheader()
        writer.writerows(seed_rows)

    with (output_dir / "exp1_vit_knn_by_k.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "k", "knn_acc"])
        writer.writeheader()
        writer.writerows(knn_rows)

    with (output_dir / "exp1_vit_linear_probe_history.json").open("w") as f:
        json.dump(histories, f, indent=2)

    print(f"\nSaved ViT Exp. 1 results to {output_dir}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Experiment 1 with tiny ViT CIFAR-100 checkpoints.")
    parser.add_argument("--ae-checkpoint", type=Path, default=Path("checkpoints/ae_tinyvit_cifar100.pth"))
    parser.add_argument("--mae-checkpoint", type=Path, default=Path("checkpoints/mae_tinyvit_cifar100.pth"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--knn-k", type=int, nargs="+", default=[5, 10, 20, 50, 100])
    parser.add_argument("--knn-train-limit", type=int, default=10000)
    parser.add_argument("--knn-test-limit", type=int, default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("results/exp1_vit_cifar100"))
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
    knn_rows = []
    histories = {}
    for name, encoder in encoders.items():
        histories[name] = {}
        for seed in args.seeds:
            set_seed(seed)
            train_loader, test_loader = build_loaders(
                args.batch_size,
                args.num_workers,
                args.max_train_samples,
                args.max_test_samples,
                seed,
            )
            linear_acc, history = run_linear_probe(
                name,
                encoder,
                train_loader,
                test_loader,
                device,
                args.epochs,
                seed,
                feature_dim,
            )
            seed_rows.append({"model": name, "seed": seed, "linear_probe_acc": linear_acc})
            histories[name][str(seed)] = history

        knn_train_loader, knn_test_loader = build_loaders(
            args.batch_size,
            args.num_workers,
            args.max_train_samples,
            args.max_test_samples,
            args.seeds[0],
        )
        knn_rows.extend(
            run_knn(
                name,
                encoder,
                knn_train_loader,
                knn_test_loader,
                device,
                args.knn_k,
                args.knn_train_limit,
                args.knn_test_limit,
            )
        )

    summary_rows = []
    for name in encoders:
        linear_values = [row["linear_probe_acc"] for row in seed_rows if row["model"] == name]
        linear_mean, linear_std = mean_std(linear_values)
        model_knn_rows = [row for row in knn_rows if row["model"] == name]
        best_knn = max(model_knn_rows, key=lambda row: row["knn_acc"])
        summary_rows.append(
            {
                "model": name,
                "linear_probe_mean": linear_mean,
                "linear_probe_std": linear_std,
                "linear_probe_seeds": len(args.seeds),
                "best_knn_k": best_knn["k"],
                "best_knn_acc": best_knn["knn_acc"],
            }
        )

    save_results(summary_rows, seed_rows, knn_rows, histories, args.output_dir)


if __name__ == "__main__":
    main()
