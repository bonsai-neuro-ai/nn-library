import warnings
from pathlib import Path
from typing import Literal, Optional

import numpy as np
from scipy.ndimage import gaussian_filter
from torch.utils.data import Dataset


class VanHateren(Dataset):
    """Dataset for the Van Hateren natural image dataset.

    Download instructions: see https://gist.github.com/wrongu/d87432950155a2a27e9c1cb85596f5e9

    Args:
        root_dir: Root directory containing the Van Hateren images.
        mode: Which set of images to load, either "iml" (linearized) or "imc" (contrast-normalized).
        local_contrast_window: If set, apply local contrast normalization with a Gaussian blur
            'sigma' set to this window size. This approximately z-scores each pixel based on local
            statistics per image. If None (default), return raw values, which are approximately in
            [0, 1] for "iml".
    """

    def __init__(
        self,
        root_dir: str | Path,
        mode: Literal["iml", "imc"] = "iml",
        local_contrast_window: Optional[float] = None,
    ):
        self.image_dir = Path(root_dir) / f"vanhateren_{mode}"
        if not self.image_dir.is_dir():
            raise ValueError(f"Directory {self.image_dir} does not exist.")
        self.filenames = list(sorted(self.image_dir.glob("*." + mode)))
        self.window = local_contrast_window

        if mode == "imc":
            warnings.warn(
                "Only use IMC if you know what you're doing. "
                "The author of this Dataset wrapper didn't test them. "
                "IMC is documented in the github repo for the vanhateren dataset."
            )

    def __getitem__(self, index):
        with open(self.filenames[index], "rb") as fin:
            img = np.frombuffer(fin.read(), dtype="uint16").byteswap().reshape(1024, 1536)
            img = img.astype(np.float32) / 4095.0

        if self.window is not None:
            # Apply local contrast normalization
            local_mean = gaussian_filter(img, sigma=self.window / 2, mode="reflect")
            local_sqr_mean = gaussian_filter(img**2, sigma=self.window / 2, mode="reflect")
            # Variance should be non-negative
            local_var = np.clip(local_sqr_mean - local_mean**2, 1e-8, None)
            img = (img - local_mean) / np.sqrt(local_var)

        assert not np.any(np.isnan(img)), "Image should not contain NaN values"
        return img

    def __len__(self):
        return len(self.filenames)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    dataset = VanHateren("/data/datasets/vanhateren")
    idx = np.random.randint(len(dataset))
    plt.subplot(2,2,1)
    plt.imshow(dataset[idx][100:200, 100:200], cmap="gray", vmin=0, vmax=1)
    plt.axis('off')

    dataset.local_contrast_window = 1
    plt.subplot(2,2,2)
    plt.imshow(dataset[idx][100:200, 100:200], cmap="gray", vmin=-2, vmax=2)
    plt.axis('off')

    dataset.local_contrast_window = 4
    plt.subplot(2,2,3)
    plt.imshow(dataset[idx][100:200, 100:200], cmap="gray", vmin=-2, vmax=2)
    plt.axis('off')

    dataset.local_contrast_window = 16
    plt.subplot(2,2,4)
    plt.imshow(dataset[idx][100:200, 100:200], cmap="gray", vmin=-2, vmax=2)
    plt.axis('off')

    plt.show()