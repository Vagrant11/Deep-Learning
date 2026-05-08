import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models import AE, MAE, ae_loss, mae_loss


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_train_loader(batch_size=128):
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=transform,
    )

    generator = torch.Generator()
    generator.manual_seed(42)

    return DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )


def train_ae(model, loader, optimizer, device):
    model.train()
    total_loss = 0

    for x, _ in loader:
        x = x.to(device)
        x_hat = model(x)
        loss = ae_loss(x_hat, x)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def train_mae(model, loader, optimizer, device):
    model.train()
    total_loss = 0

    for x, _ in loader:
        x = x.to(device)
        x_hat, x, mask = model(x)
        loss = mae_loss(x_hat, x, mask)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def save_checkpoint(model, optimizer, path):
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
        },
        path,
    )


def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = build_train_loader(batch_size=128)

    ae = AE().to(device)
    opt_ae = torch.optim.Adam(ae.parameters(), lr=1e-3)

    print("Training AE...")
    for epoch in range(5):
        loss = train_ae(ae, train_loader, opt_ae, device)
        print(f"AE Epoch {epoch}: {loss:.4f}")

    save_checkpoint(ae, opt_ae, "checkpoints/ae_cifar10.pth")
    print("AE saved to checkpoints/ae_cifar10.pth\n")

    base = AE().to(device)
    mae = MAE(base).to(device)
    opt_mae = torch.optim.Adam(mae.parameters(), lr=1e-3)

    print("Training MAE...")
    for epoch in range(5):
        loss = train_mae(mae, train_loader, opt_mae, device)
        print(f"MAE Epoch {epoch}: {loss:.4f}")

    save_checkpoint(mae, opt_mae, "checkpoints/mae_cifar10.pth")
    print("MAE saved to checkpoints/mae_cifar10.pth")


if __name__ == "__main__":
    main()
