import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from AE import AE, MAE


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================
# 数据
# =========================
transform = transforms.Compose([transforms.ToTensor()])

train_dataset = datasets.CIFAR10(root="./data", train=True,  download=True, transform=transform)
test_dataset  = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

g = torch.Generator()
g.manual_seed(42)

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True,  generator=g)
test_loader  = DataLoader(test_dataset,  batch_size=256, shuffle=False)


# =========================
# Linear Probe 模型
# encoder 冻结，只训练线性层
# =========================
class LinearProbe(nn.Module):
    def __init__(self, encoder, feature_dim=4096, num_classes=10):
        super().__init__()
        self.encoder = encoder
        for param in self.encoder.parameters():
            param.requires_grad = False   # 冻结 encoder

        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        with torch.no_grad():
            z = self.encoder(x)           # (B, 256, 4, 4)
        z = z.view(z.size(0), -1)         # (B, 4096)
        return self.classifier(z)


# =========================
# 训练 / 评估
# =========================
def train_epoch(model, loader, optimizer):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)
    return total_loss / len(loader), correct / total


def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(1) == y).sum().item()
            total += y.size(0)
    return correct / total


def run_probe(name, encoder, epochs=10):
    print(f"\n=== Linear Probe: {name} ===")
    probe = LinearProbe(encoder).to(device)
    optimizer = torch.optim.Adam(probe.classifier.parameters(), lr=1e-3)

    for epoch in range(epochs):
        loss, train_acc = train_epoch(probe, train_loader, optimizer)
        test_acc = evaluate(probe, test_loader)
        print(f"Epoch {epoch}: loss={loss:.4f}  train={train_acc:.4f}  test={test_acc:.4f}")

    print(f"{name} 最终 test accuracy: {test_acc:.4f}")
    return test_acc


# =========================
# 加载权重，提取 encoder
# =========================
if __name__ == "__main__":

    # --- AE encoder ---
    ae = AE().to(device)
    ckpt = torch.load("ae_cifar10.pth", map_location=device)
    ae.load_state_dict(ckpt["model_state"])
    ae_encoder = ae.encoder

    # --- MAE encoder ---
    base = AE().to(device)
    mae = MAE(base).to(device)
    ckpt = torch.load("mae_cifar10.pth", map_location=device)
    mae.load_state_dict(ckpt["model_state"])
    mae_encoder = mae.encoder

    # --- 跑 Linear Probe ---
    acc_ae  = run_probe("AE",  ae_encoder,  epochs=10)
    acc_mae = run_probe("MAE", mae_encoder, epochs=10)

    print(f"\n{'='*40}")
    print(f"AE  test accuracy: {acc_ae:.4f}")
    print(f"MAE test accuracy: {acc_mae:.4f}")
    print(f"MAE 比 AE 高: {(acc_mae - acc_ae)*100:.2f}%")
