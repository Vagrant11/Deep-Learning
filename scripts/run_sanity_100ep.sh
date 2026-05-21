#!/usr/bin/env bash
set -euo pipefail

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY

python scripts/pretrain_vit_mae_cifar100.py \
  --epochs 100 \
  --batch-size 256 \
  --num-workers 4 \
  --amp \
  --output-dir checkpoints/sanity_100ep

python scripts/run_exp1_vit_cifar100.py \
  --ae-checkpoint checkpoints/sanity_100ep/ae_tinyvit_cifar100.pth \
  --mae-checkpoint checkpoints/sanity_100ep/mae_tinyvit_cifar100.pth \
  --epochs 10 \
  --seeds 42 \
  --batch-size 256 \
  --num-workers 4 \
  --skip-knn \
  --output-dir results/sanity_100ep

python scripts/run_exp2_vit_cifar100.py \
  --ae-checkpoint checkpoints/sanity_100ep/ae_tinyvit_cifar100.pth \
  --mae-checkpoint checkpoints/sanity_100ep/mae_tinyvit_cifar100.pth \
  --exp1-summary results/sanity_100ep/exp1_vit_summary.csv \
  --batch-size 256 \
  --num-workers 4 \
  --output-dir results/sanity_100ep_exp2

echo
echo "Sanity-check linear probe summary:"
cat results/sanity_100ep/exp1_vit_summary.csv
echo
echo "Sanity-check reconstruction summary:"
cat results/sanity_100ep_exp2/exp2_vit_reconstruction_vs_representation.csv
