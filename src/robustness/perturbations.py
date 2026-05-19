import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


def high_pass_filter(x, kernel_size=5, strength=1.0):
    """Suppress low-frequency content while keeping the output in [0, 1]."""
    if x.dim() == 3:
        x = x.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False

    blurred = F.avg_pool2d(x, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
    high = x - blurred
    output = 0.5 + strength * high
    output = output.clamp(0.0, 1.0)
    return output.squeeze(0) if squeeze else output


def patch_shuffle(x, patch_size=4, generator=None):
    """Shuffle non-overlapping patches while preserving each patch's pixels."""
    if x.dim() == 3:
        x = x.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False

    batch_size, channels, height, width = x.shape
    if height % patch_size != 0 or width % patch_size != 0:
        raise ValueError("Image height and width must be divisible by patch_size")

    grid_h = height // patch_size
    grid_w = width // patch_size
    patches = x.reshape(batch_size, channels, grid_h, patch_size, grid_w, patch_size)
    patches = patches.permute(0, 2, 4, 1, 3, 5).reshape(batch_size, grid_h * grid_w, channels, patch_size, patch_size)

    shuffled = []
    for index in range(batch_size):
        order = torch.randperm(grid_h * grid_w, device=x.device, generator=generator)
        shuffled.append(patches[index, order])
    patches = torch.stack(shuffled, dim=0)

    output = patches.reshape(batch_size, grid_h, grid_w, channels, patch_size, patch_size)
    output = output.permute(0, 3, 1, 4, 2, 5).reshape(batch_size, channels, height, width)
    return output.squeeze(0) if squeeze else output


def patch_drop(x, patch_size=4, drop_ratio=0.5, fill_value=0.5, generator=None):
    """Replace a random subset of patches with a constant value."""
    if x.dim() == 3:
        x = x.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False

    batch_size, channels, height, width = x.shape
    if height % patch_size != 0 or width % patch_size != 0:
        raise ValueError("Image height and width must be divisible by patch_size")

    grid_h = height // patch_size
    grid_w = width // patch_size
    num_patches = grid_h * grid_w
    num_dropped = int(num_patches * drop_ratio)

    output = x.clone()
    for index in range(batch_size):
        order = torch.randperm(num_patches, device=x.device, generator=generator)[:num_dropped]
        rows = order // grid_w
        cols = order % grid_w
        for row, col in zip(rows.tolist(), cols.tolist()):
            y0 = row * patch_size
            x0 = col * patch_size
            output[index, :, y0 : y0 + patch_size, x0 : x0 + patch_size] = fill_value

    return output.squeeze(0) if squeeze else output


class PerturbedDataset(Dataset):
    def __init__(
        self,
        dataset,
        perturbation,
        patch_size=4,
        drop_ratio=0.5,
        high_pass_kernel=5,
        high_pass_strength=1.0,
        seed=42,
    ):
        self.dataset = dataset
        self.perturbation = perturbation
        self.patch_size = patch_size
        self.drop_ratio = drop_ratio
        self.high_pass_kernel = high_pass_kernel
        self.high_pass_strength = high_pass_strength
        self.seed = seed

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, label = self.dataset[index]
        generator = torch.Generator(device=image.device)
        generator.manual_seed(self.seed + index)

        if self.perturbation == "clean":
            return image, label
        if self.perturbation == "high_pass":
            return high_pass_filter(image, self.high_pass_kernel, self.high_pass_strength), label
        if self.perturbation == "patch_shuffle":
            return patch_shuffle(image, self.patch_size, generator), label
        if self.perturbation == "patch_drop":
            return patch_drop(image, self.patch_size, self.drop_ratio, generator=generator), label

        raise ValueError(f"Unknown perturbation: {self.perturbation}")
