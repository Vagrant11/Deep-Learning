import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models import TinyViTReconstructor, reconstruction_loss


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_train_loader(batch_size, num_workers, max_train_samples, seed):
    transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )
    dataset = datasets.CIFAR100(root="./data", train=True, download=True, transform=transform)
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


def build_model(args):
    return TinyViTReconstructor(
        image_size=32,
        patch_size=args.patch_size,
        embed_dim=args.embed_dim,
        encoder_depth=args.encoder_depth,
        decoder_depth=args.decoder_depth,
        num_heads=args.num_heads,
    )


def train_one_epoch(model, loader, optimizer, scaler, device, mask_ratio, use_amp):
    model.train()
    total_loss = 0.0

    for x, _ in loader:
        x = x.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            pred_patches, target_patches, mask, _ = model(x, mask_ratio=mask_ratio)
            loss = reconstruction_loss(pred_patches, target_patches, mask)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    return total_loss / len(loader)


def save_checkpoint(model, optimizer, args, model_type, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_type": model_type,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "args": vars(args),
        },
        path,
    )


def pretrain_model(model_type, mask_ratio, args, train_loader, device):
    model = build_model(args).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    use_amp = args.amp and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    print(f"\n=== Pretraining {model_type} tiny ViT, mask_ratio={mask_ratio} ===", flush=True)
    for epoch in range(args.epochs):
        loss = train_one_epoch(model, train_loader, optimizer, scaler, device, mask_ratio, use_amp)
        print(f"{model_type} epoch {epoch}: loss={loss:.5f}", flush=True)

    ckpt_path = args.output_dir / f"{model_type.lower()}_tinyvit_cifar100.pth"
    save_checkpoint(model, optimizer, args, model_type, ckpt_path)
    print(f"Saved {model_type} checkpoint to {ckpt_path}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Pretrain tiny ViT AE/MAE on CIFAR-100.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--patch-size", type=int, default=4)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--encoder-depth", type=int, default=4)
    parser.add_argument("--decoder-depth", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--mae-mask-ratio", type=float, default=0.75)

    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints"))
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)
    print(f"AMP enabled: {args.amp and device.type == 'cuda'}", flush=True)

    train_loader = build_train_loader(
        args.batch_size,
        args.num_workers,
        args.max_train_samples,
        args.seed,
    )

    pretrain_model("AE", 0.0, args, train_loader, device)
    pretrain_model("MAE", args.mae_mask_ratio, args, train_loader, device)


if __name__ == "__main__":
    main()
