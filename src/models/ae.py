import torch
import torch.nn as nn
import torch.nn.functional as F


class AE(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 4, 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


def random_mask(x, mask_ratio=0.75):
    batch_size, _, height, width = x.shape
    mask = torch.rand(batch_size, 1, height, width, device=x.device)
    mask = (mask > mask_ratio).float()
    return x * mask, mask


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


def ae_loss(x_hat, x):
    return F.mse_loss(x_hat, x)


def mae_loss(x_hat, x, mask):
    return ((x_hat - x) ** 2 * (1 - mask)).mean()
