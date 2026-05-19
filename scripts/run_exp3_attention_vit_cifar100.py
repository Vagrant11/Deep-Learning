import argparse
import csv
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.attention import encoder_attention_distances, effective_receptive_field
from src.models import TinyViTReconstructor


def build_test_loader(batch_size, num_workers, max_test_samples):
    dataset = datasets.CIFAR100(
        root="./data",
        train=False,
        download=True,
        transform=transforms.ToTensor(),
    )
    if max_test_samples is not None:
        dataset = Subset(dataset, list(range(max_test_samples)))

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


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


def _accumulate_list(target, key, values, batch_size):
    if key not in target:
        target[key] = [0.0 for _ in values]
    for index, value in enumerate(values):
        target[key][index] += value * batch_size


def _accumulate_scalar(target, key, value, batch_size):
    if value == "":
        return
    target[key] = target.get(key, 0.0) + value * batch_size


def average_attention_rows(model, loader, device, mask_ratio, far_threshold):
    scalar_sums = {}
    layer_head_sums = {}
    num_batches = 0
    num_images = 0

    for x, _ in loader:
        x = x.to(device, non_blocking=True)
        rows, _ = encoder_attention_distances(model, x, mask_ratio=mask_ratio, far_threshold=far_threshold)
        batch_size = x.size(0)

        for row in rows:
            layer = row["layer"]
            for key in [
                "mean_attention_distance",
                "far_attention_mass",
                "masked_to_visible_distance",
                "visible_to_visible_distance",
            ]:
                _accumulate_scalar(scalar_sums, (layer, key), row[key], batch_size)
            for key in [
                "head_distances",
                "head_far_attention_mass",
                "head_masked_to_visible_distances",
                "head_visible_to_visible_distances",
            ]:
                _accumulate_list(layer_head_sums, (layer, key), row[key], batch_size)

        num_batches += 1
        num_images += batch_size
        print(f"  processed attention batch {num_batches}, images={num_images}", flush=True)

    output = []
    layers = sorted({layer for layer, _ in scalar_sums.keys()})
    for layer in layers:
        output.append(
            {
                "layer": layer,
                "mean_attention_distance": scalar_sums[(layer, "mean_attention_distance")] / num_images,
                "far_attention_mass": scalar_sums[(layer, "far_attention_mass")] / num_images,
                "masked_to_visible_distance": (
                    scalar_sums.get((layer, "masked_to_visible_distance"), "")
                    if (layer, "masked_to_visible_distance") not in scalar_sums
                    else scalar_sums[(layer, "masked_to_visible_distance")] / num_images
                ),
                "visible_to_visible_distance": (
                    scalar_sums.get((layer, "visible_to_visible_distance"), "")
                    if (layer, "visible_to_visible_distance") not in scalar_sums
                    else scalar_sums[(layer, "visible_to_visible_distance")] / num_images
                ),
                "head_distances": [value / num_images for value in layer_head_sums[(layer, "head_distances")]],
                "head_far_attention_mass": [
                    value / num_images for value in layer_head_sums[(layer, "head_far_attention_mass")]
                ],
                "head_masked_to_visible_distances": [
                    value / num_images for value in layer_head_sums.get((layer, "head_masked_to_visible_distances"), [])
                ],
                "head_visible_to_visible_distances": [
                    value / num_images for value in layer_head_sums.get((layer, "head_visible_to_visible_distances"), [])
                ],
            }
        )
    return output, num_images


def average_erf(model, loader, device, max_batches):
    metric_sums = {}
    num_batches = 0
    num_images = 0

    for x, _ in loader:
        x = x.to(device, non_blocking=True)
        metrics = effective_receptive_field(model, x)
        batch_size = x.size(0)
        for key, value in metrics.items():
            metric_sums[key] = metric_sums.get(key, 0.0) + value * batch_size
        num_batches += 1
        num_images += batch_size
        print(f"  processed ERF batch {num_batches}, images={num_images}", flush=True)
        if max_batches is not None and num_batches >= max_batches:
            break

    return {key: value / num_images for key, value in metric_sums.items()}, num_images


def save_results(attention_rows, erf_rows, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "exp3_attention_distance_by_layer.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "condition",
                "mask_ratio",
                "layer",
                "mean_attention_distance",
                "far_attention_mass",
                "masked_to_visible_distance",
                "visible_to_visible_distance",
                "num_images",
                "head_distances_json",
                "head_far_attention_mass_json",
                "head_masked_to_visible_distances_json",
                "head_visible_to_visible_distances_json",
            ],
        )
        writer.writeheader()
        writer.writerows(attention_rows)

    with (output_dir / "exp3_erf_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "num_images",
                "erf_radius_px",
                "erf_entropy",
                "erf_area_50_frac",
                "erf_area_90_frac",
                "gradient_abs_mean",
            ],
        )
        writer.writeheader()
        writer.writerows(erf_rows)

    print(f"\nSaved ViT Exp. 3 results to {output_dir}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Experiment 3: attention distance and ERF for tiny ViT CIFAR-100.")
    parser.add_argument("--ae-checkpoint", type=Path, default=Path("checkpoints/ae_tinyvit_cifar100.pth"))
    parser.add_argument("--mae-checkpoint", type=Path, default=Path("checkpoints/mae_tinyvit_cifar100.pth"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-test-samples", type=int, default=500)
    parser.add_argument("--erf-max-batches", type=int, default=None)
    parser.add_argument("--mae-mask-ratio", type=float, default=0.75)
    parser.add_argument("--far-threshold", type=float, default=4.0)
    parser.add_argument("--skip-masked-mae", action="store_true")
    parser.add_argument("--skip-erf", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("results/exp3_attention"))
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    loader = build_test_loader(args.batch_size, args.num_workers, args.max_test_samples)
    configs = [
        ("AE", args.ae_checkpoint, [("full_image", 0.0)]),
        (
            "MAE",
            args.mae_checkpoint,
            [("full_image", 0.0)]
            if args.skip_masked_mae
            else [("full_image", 0.0), ("masked_reconstruction", args.mae_mask_ratio)],
        ),
    ]

    attention_output = []
    erf_output = []
    for name, checkpoint, attention_conditions in configs:
        print(f"\n=== ViT Exp. 3: {name} ===", flush=True)
        model = load_reconstructor(checkpoint, device)

        for condition, mask_ratio in attention_conditions:
            print(f"\n--- attention condition: {condition}, mask_ratio={mask_ratio} ---", flush=True)
            layer_rows, attention_images = average_attention_rows(
                model, loader, device, mask_ratio, args.far_threshold
            )
            for row in layer_rows:
                attention_output.append(
                    {
                        "model": name,
                        "condition": condition,
                        "mask_ratio": mask_ratio,
                        "layer": row["layer"],
                        "mean_attention_distance": row["mean_attention_distance"],
                        "far_attention_mass": row["far_attention_mass"],
                        "masked_to_visible_distance": row["masked_to_visible_distance"],
                        "visible_to_visible_distance": row["visible_to_visible_distance"],
                        "num_images": attention_images,
                        "head_distances_json": json.dumps(row["head_distances"]),
                        "head_far_attention_mass_json": json.dumps(row["head_far_attention_mass"]),
                        "head_masked_to_visible_distances_json": json.dumps(
                            row["head_masked_to_visible_distances"]
                        ),
                        "head_visible_to_visible_distances_json": json.dumps(
                            row["head_visible_to_visible_distances"]
                        ),
                    }
                )
                extra = ""
                if row["masked_to_visible_distance"] != "":
                    extra = f", masked_to_visible={row['masked_to_visible_distance']:.4f}"
                print(
                    f"{name} {condition} layer {row['layer']}: "
                    f"mean_attention_distance={row['mean_attention_distance']:.4f}, "
                    f"far_mass={row['far_attention_mass']:.4f}{extra}",
                    flush=True,
                )

        if not args.skip_erf:
            erf_metrics, erf_images = average_erf(model, loader, device, args.erf_max_batches)
            erf_output.append({"model": name, "num_images": erf_images, **erf_metrics})
            print(
                f"{name} ERF: radius={erf_metrics['erf_radius_px']:.4f}px, "
                f"entropy={erf_metrics['erf_entropy']:.4f}",
                flush=True,
            )

    if args.skip_erf:
        erf_output = [
            {
                "model": name,
                "num_images": 0,
                "erf_radius_px": "",
                "erf_entropy": "",
                "erf_area_50_frac": "",
                "erf_area_90_frac": "",
                "gradient_abs_mean": "",
            }
            for name, _, _ in configs
        ]

    save_results(attention_output, erf_output, args.output_dir)


if __name__ == "__main__":
    main()
