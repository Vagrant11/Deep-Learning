import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models import AE, MAE
from src.probing.knn import extract_features, knn_accuracy
from src.probing.linear_probe import LinearProbe, evaluate, train_epoch


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_loaders(batch_size):
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

    generator = torch.Generator()
    generator.manual_seed(42)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=generator)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def load_encoders(device):
    ae = AE().to(device)
    ckpt = torch.load("checkpoints/ae_cifar10.pth", map_location=device)
    ae.load_state_dict(ckpt["model_state"])

    base = AE().to(device)
    mae = MAE(base).to(device)
    ckpt = torch.load("checkpoints/mae_cifar10.pth", map_location=device)
    mae.load_state_dict(ckpt["model_state"])

    return {"AE": ae.encoder, "MAE": mae.encoder}


def run_linear_probe(name, encoder, train_loader, test_loader, device, epochs):
    print(f"\n=== Exp. 1 Linear Probe: {name} ===", flush=True)
    probe = LinearProbe(encoder).to(device)
    optimizer = torch.optim.Adam(probe.classifier.parameters(), lr=1e-3)

    history = []
    test_acc = 0
    for epoch in range(epochs):
        loss, train_acc = train_epoch(probe, train_loader, optimizer, device)
        test_acc = evaluate(probe, test_loader, device)
        history.append(
            {
                "epoch": epoch,
                "loss": loss,
                "train_acc": train_acc,
                "test_acc": test_acc,
            }
        )
        print(f"Epoch {epoch}: loss={loss:.4f}  train={train_acc:.4f}  test={test_acc:.4f}", flush=True)

    return test_acc, history


def run_knn(name, encoder, train_loader, test_loader, device, k, train_limit, test_limit):
    print(f"\n=== Exp. 1 k-NN: {name} ===", flush=True)
    print(f"Extracting train features, limit={train_limit or 'all'}", flush=True)
    train_features, train_labels = extract_features(encoder, train_loader, device, train_limit)
    print(f"Extracting test features, limit={test_limit or 'all'}", flush=True)
    test_features, test_labels = extract_features(encoder, test_loader, device, test_limit)

    acc = knn_accuracy(train_features, train_labels, test_features, test_labels, k=k)
    print(f"{name} k-NN accuracy, k={k}: {acc:.4f}", flush=True)
    return acc


def save_results(results, histories, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "exp1_summary.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "linear_probe_acc", "knn_acc"])
        writer.writeheader()
        writer.writerows(results)

    json_path = output_dir / "exp1_linear_probe_history.json"
    with json_path.open("w") as f:
        json.dump(histories, f, indent=2)

    print(f"\nSaved summary to {csv_path}", flush=True)
    print(f"Saved linear probe history to {json_path}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Experiment 1: linear probing and k-NN.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--knn-k", type=int, default=20)
    parser.add_argument("--knn-train-limit", type=int, default=10000)
    parser.add_argument("--knn-test-limit", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("results/exp1"))
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    train_loader, test_loader = build_loaders(args.batch_size)
    encoders = load_encoders(device)

    results = []
    histories = {}
    for name, encoder in encoders.items():
        linear_acc, history = run_linear_probe(
            name,
            encoder,
            train_loader,
            test_loader,
            device,
            args.epochs,
        )
        knn_acc = run_knn(
            name,
            encoder,
            train_loader,
            test_loader,
            device,
            args.knn_k,
            args.knn_train_limit,
            args.knn_test_limit,
        )

        results.append(
            {
                "model": name,
                "linear_probe_acc": linear_acc,
                "knn_acc": knn_acc,
            }
        )
        histories[name] = history

    save_results(results, histories, args.output_dir)


if __name__ == "__main__":
    main()
