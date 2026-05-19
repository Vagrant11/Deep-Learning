import torch
from torch.utils.data import Dataset


def fourier_shape_texture_mix(shape_image, texture_image, alpha=1.0):
    """Combine shape phase with texture amplitude in the Fourier domain.

    The phase of an image tends to preserve spatial layout, while the amplitude
    contains frequency-energy statistics that are a useful lightweight proxy for
    texture. Inputs and outputs are tensors shaped C x H x W in [0, 1].
    """
    shape_fft = torch.fft.fft2(shape_image, dim=(-2, -1))
    texture_fft = torch.fft.fft2(texture_image, dim=(-2, -1))

    shape_amp = torch.abs(shape_fft)
    texture_amp = torch.abs(texture_fft)
    mixed_amp = (1.0 - alpha) * shape_amp + alpha * texture_amp
    mixed_fft = mixed_amp * torch.exp(1j * torch.angle(shape_fft))
    mixed = torch.fft.ifft2(mixed_fft, dim=(-2, -1)).real

    min_value = mixed.amin(dim=(-2, -1), keepdim=True)
    max_value = mixed.amax(dim=(-2, -1), keepdim=True)
    mixed = (mixed - min_value) / (max_value - min_value).clamp(min=1e-6)
    return mixed.clamp(0.0, 1.0)


class FourierConflictDataset(Dataset):
    def __init__(self, dataset, alpha=1.0, offset=137):
        self.dataset = dataset
        self.alpha = alpha
        self.offset = offset
        self.texture_indices = self._build_texture_indices()

    def _build_texture_indices(self):
        labels = [self.dataset[index][1] for index in range(len(self.dataset))]
        texture_indices = []
        num_items = len(labels)

        for index, shape_label in enumerate(labels):
            candidate = (index + self.offset) % num_items
            while labels[candidate] == shape_label:
                candidate = (candidate + 1) % num_items
            texture_indices.append(candidate)

        return texture_indices

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        shape_image, shape_label = self.dataset[index]
        texture_image, texture_label = self.dataset[self.texture_indices[index]]
        mixed = fourier_shape_texture_mix(shape_image, texture_image, alpha=self.alpha)
        return mixed, shape_label, texture_label
