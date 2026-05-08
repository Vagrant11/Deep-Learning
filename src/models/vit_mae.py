import torch
import torch.nn as nn
import torch.nn.functional as F


class TinyViTReconstructor(nn.Module):
    def __init__(
        self,
        image_size=32,
        patch_size=4,
        in_channels=3,
        embed_dim=128,
        encoder_depth=4,
        decoder_depth=2,
        num_heads=4,
        mlp_ratio=4.0,
    ):
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")

        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.num_patches_per_side = image_size // patch_size
        self.num_patches = self.num_patches_per_side ** 2
        self.patch_dim = in_channels * patch_size * patch_size
        self.embed_dim = embed_dim

        self.patch_embed = nn.Linear(self.patch_dim, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=encoder_depth)

        decoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=decoder_depth)
        self.pred = nn.Linear(embed_dim, self.patch_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.xavier_uniform_(self.patch_embed.weight)
        nn.init.zeros_(self.patch_embed.bias)
        nn.init.xavier_uniform_(self.pred.weight)
        nn.init.zeros_(self.pred.bias)

    def patchify(self, x):
        batch_size, channels, height, width = x.shape
        patches = x.reshape(
            batch_size,
            channels,
            self.num_patches_per_side,
            self.patch_size,
            self.num_patches_per_side,
            self.patch_size,
        )
        patches = patches.permute(0, 2, 4, 1, 3, 5)
        return patches.reshape(batch_size, self.num_patches, self.patch_dim)

    def unpatchify(self, patches):
        batch_size = patches.size(0)
        x = patches.reshape(
            batch_size,
            self.num_patches_per_side,
            self.num_patches_per_side,
            self.in_channels,
            self.patch_size,
            self.patch_size,
        )
        x = x.permute(0, 3, 1, 4, 2, 5)
        return x.reshape(batch_size, self.in_channels, self.image_size, self.image_size)

    def random_mask(self, batch_size, mask_ratio, device):
        num_masked = int(self.num_patches * mask_ratio)
        noise = torch.rand(batch_size, self.num_patches, device=device)
        ids_shuffle = torch.argsort(noise, dim=1)
        mask = torch.zeros(batch_size, self.num_patches, device=device)
        mask.scatter_(1, ids_shuffle[:, :num_masked], 1.0)
        return mask

    def encode_tokens(self, x, mask_ratio=0.0):
        patches = self.patchify(x)
        tokens = self.patch_embed(patches)

        mask = None
        if mask_ratio > 0:
            mask = self.random_mask(x.size(0), mask_ratio, x.device)
            mask_tokens = self.mask_token.expand(x.size(0), self.num_patches, -1)
            tokens = torch.where(mask.unsqueeze(-1).bool(), mask_tokens, tokens)

        tokens = tokens + self.pos_embed
        encoded = self.encoder(tokens)
        return encoded, patches, mask

    def encode_features(self, x):
        encoded, _, _ = self.encode_tokens(x, mask_ratio=0.0)
        return encoded.mean(dim=1)

    def forward(self, x, mask_ratio=0.0):
        encoded, target_patches, mask = self.encode_tokens(x, mask_ratio=mask_ratio)
        decoded = self.decoder(encoded)
        pred_patches = self.pred(decoded)
        recon = self.unpatchify(pred_patches).sigmoid()
        return pred_patches, target_patches, mask, recon


def reconstruction_loss(pred_patches, target_patches, mask=None):
    patch_loss = F.mse_loss(pred_patches, target_patches, reduction="none").mean(dim=-1)
    if mask is None:
        return patch_loss.mean()

    denom = mask.sum().clamp(min=1.0)
    return (patch_loss * mask).sum() / denom


class FeatureEncoder(nn.Module):
    def __init__(self, reconstructor):
        super().__init__()
        self.reconstructor = reconstructor

    def forward(self, x):
        return self.reconstructor.encode_features(x)
