import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# =========================
# 0. Reproducibility（关键）
# =========================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================
# 1. Data (CIFAR-10)
# =========================
transform = transforms.Compose([
    transforms.ToTensor()
])

train_dataset = datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

g = torch.Generator()
g.manual_seed(42)

train_loader = DataLoader(
    train_dataset,
    batch_size=128,
    shuffle=True,
    generator=g
)


# =========================
# 2. AE Model (shared backbone)
# =========================
class AE(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),   # 32 → 16
            nn.ReLU(),

            nn.Conv2d(64, 128, 4, 2, 1), # 16 → 8
            nn.ReLU(),

            nn.Conv2d(128, 256, 4, 2, 1), # 8 → 4
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.ReLU(),

            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(),

            nn.ConvTranspose2d(64, 3, 4, 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat


# =========================
# 3. MAE masking
# =========================
def random_mask(x, mask_ratio=0.75):
    B, C, H, W = x.shape
    mask = torch.rand(B, 1, H, W, device=x.device)
    mask = (mask > mask_ratio).float()
    return x * mask, mask


# =========================
# 4. MAE wrapper
# =========================
class MAE(nn.Module):
    def __init__(self, ae):
        super().__init__()
        self.encoder = ae.encoder
        self.decoder = ae.decoder

    def forward(self, x):
        x_masked, mask = random_mask(x, 0.75)
        z = self.encoder(x_masked)
        x_hat = self.decoder(z)
        return x_hat, x, mask


# =========================
# 5. Loss functions
# =========================
def ae_loss(x_hat, x):
    return F.mse_loss(x_hat, x)

def mae_loss(x_hat, x, mask):
    return ((x_hat - x) ** 2 * (1 - mask)).mean()


# =========================
# 6. Training AE
# =========================
def train_ae(model, loader, optimizer):
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


# =========================
# 7. Training MAE
# =========================
def train_mae(model, loader, optimizer):
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


# =========================
# 8. Save function
# =========================
def save_checkpoint(model, optimizer, path):
    torch.save({
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict()
    }, path)


# =========================
# 9. Run experiment
# =========================
if __name__ == "__main__":

    # -------- AE --------
    ae = AE().to(device)
    opt_ae = torch.optim.Adam(ae.parameters(), lr=1e-3)

    print("Training AE...")
    for epoch in range(5):
        loss = train_ae(ae, train_loader, opt_ae)
        print(f"AE Epoch {epoch}: {loss:.4f}")

    save_checkpoint(ae, opt_ae, "ae_cifar10.pth")
    print("AE saved.\n")


    # -------- MAE --------
    base = AE().to(device)
    mae = MAE(base).to(device)
    opt_mae = torch.optim.Adam(mae.parameters(), lr=1e-3)

    print("Training MAE...")
    for epoch in range(5):
        loss = train_mae(mae, train_loader, opt_mae)
        print(f"MAE Epoch {epoch}: {loss:.4f}")

    save_checkpoint(mae, opt_mae, "mae_cifar10.pth")
    print("MAE saved.")