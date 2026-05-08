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
from src.probing.linear_probe import LinearProbe, evaluate, train_epoch


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_loaders(batch_size=256):
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

    generator = torch.Generator()
    generator.manual_seed(42)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=generator)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def run_probe(name, encoder, train_loader, test_loader, device, epochs=10):
    print(f"\n=== Linear Probe: {name} ===")
    probe = LinearProbe(encoder).to(device)
    optimizer = torch.optim.Adam(probe.classifier.parameters(), lr=1e-3)

    test_acc = 0
    for epoch in range(epochs):
        loss, train_acc = train_epoch(probe, train_loader, optimizer, device)
        test_acc = evaluate(probe, test_loader, device)
        print(f"Epoch {epoch}: loss={loss:.4f}  train={train_acc:.4f}  test={test_acc:.4f}")

    print(f"{name} final test accuracy: {test_acc:.4f}")
    return test_acc


def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = build_loaders(batch_size=256)

    ae = AE().to(device)
    ckpt = torch.load("checkpoints/ae_cifar10.pth", map_location=device)
    ae.load_state_dict(ckpt["model_state"])

    base = AE().to(device)
    mae = MAE(base).to(device)
    ckpt = torch.load("checkpoints/mae_cifar10.pth", map_location=device)
    mae.load_state_dict(ckpt["model_state"])

    acc_ae = run_probe("AE", ae.encoder, train_loader, test_loader, device, epochs=10)
    acc_mae = run_probe("MAE", mae.encoder, train_loader, test_loader, device, epochs=10)

    print("\n" + "=" * 40)
    print(f"AE  test accuracy: {acc_ae:.4f}")
    print(f"MAE test accuracy: {acc_mae:.4f}")
    print(f"MAE - AE: {(acc_mae - acc_ae) * 100:.2f}%")


if __name__ == "__main__":
    main()
