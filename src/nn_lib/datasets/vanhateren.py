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
            'sigma' set to half this window size. This approximately z-scores each pixel based on local
            statistics per image. If None (default), return raw values, which are approximately in
            [0, 1] but may be outside that range.
        zscore_epsilon: When doing local contrast normalization, this is the minimum variance to use
            when dividing by the local standard deviation. This prevents amplifying noise. Has no
            effect if local_contrast_window is None.
        clip: optional tuple specifying (low, hi) values to clip to after all other processing
            is done. This can mitigate outliers created by z-scoring**. If local_contrast_window is
            None so z-scoring is not applied, clipping to [0, 1] is reasonable. If
            local_contrast_window is set and z-scoring is applied, clipping to [-5, +5] is
            reasonable.
        crop_border: If > 0, crop this many pixels from each border of the image after all other
            processing. This is useful to avoid edge artifacts from local contrast normalization.

        **Note: setting a large local_contrast_window can lead to extreme outliers after z-scoring,
        which can cause some issues. Example: a local region is zoomed in on overlapping tree
        branches where all pixels except one are dark. The local mean is dark and the local variance
        is small, so the one bright pixel of light poking through the tree gets dramatically
        amplified. Outliers like this get more extreme as the local_contrast_window increases.
    """

    def __init__(
        self,
        root_dir: str | Path,
        mode: Literal["iml", "imc"] = "iml",
        local_contrast_window: Optional[float] = None,
        zscore_epsilon: float = 1e-3,
        clip: tuple[float | None, float | None] = (None, None),
        crop_border: int = 0,
    ):
        self.image_dir = Path(root_dir) / f"vanhateren_{mode}"
        if not self.image_dir.is_dir():
            raise ValueError(f"Directory {self.image_dir} does not exist.")
        self.filenames = list(sorted(self.image_dir.glob("*." + mode)))
        self.window = local_contrast_window
        self.zscore_epsilon = zscore_epsilon
        self.clip = clip
        self.crop_border = crop_border

    def __getitem__(self, index):
        with open(self.filenames[index], "rb") as fin:
            img = np.frombuffer(fin.read(), dtype="uint16").byteswap().reshape(1024, 1536)
            # original paper + dataset github say the 'effective' bit depth is 12, so we divide
            # by 2^12 here with the understanding that this only 'effectively' or 'mostly' puts
            # values in the [0, 1] range.
            img = img.astype(np.float32) / 4095.0
            # from the original paper: downsample by a factor of 2 to "reduce aliasing artifacts"
            img = (img[0::2, 0::2] + img[1::2, 0::2] + img[0::2, 1::2] + img[1::2, 1::2]) / 4

        # If 'window' is set, we do z-scoring
        if self.window is not None:
            local_mean = gaussian_filter(img, sigma=self.window / 2, mode="reflect")
            local_sqr_mean = gaussian_filter(img**2, sigma=self.window / 2, mode="reflect")
            # Clip the variance to avoid division by zero / amplifying noise in 'flat' regions
            local_var = np.clip(local_sqr_mean - local_mean**2, self.zscore_epsilon, None)
            img = (img - local_mean) / np.sqrt(local_var)

        # If a clipping is set, apply it after all other processing. This can mitigate extreme
        # outliers created by z-scoring or values outside the 'effective' 12-bit depth.
        if self.clip[0] is not None or self.clip[1] is not None:
            img = np.clip(img, *self.clip)

        if self.crop_border:
            img = img[self.window : -self.window, self.window : -self.window]

        assert not np.any(np.isnan(img)), "Image should not contain NaN values"
        return img

    def __len__(self):
        return len(self.filenames)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    dataset = VanHateren("/data/datasets/vanhateren", mode="imc")
    idx = np.random.randint(len(dataset))
    plt.subplot(2, 2, 1)
    plt.imshow(dataset[idx][100:200, 100:200], cmap="gray", vmin=0, vmax=1)
    plt.axis("off")

    dataset.local_contrast_window = 1
    plt.subplot(2, 2, 2)
    plt.imshow(dataset[idx][100:200, 100:200], cmap="gray", vmin=-3, vmax=3)
    plt.axis("off")

    dataset.local_contrast_window = 4
    plt.subplot(2, 2, 3)
    plt.imshow(dataset[idx][100:200, 100:200], cmap="gray", vmin=-3, vmax=3)
    plt.axis("off")

    dataset.local_contrast_window = 16
    plt.subplot(2, 2, 4)
    plt.imshow(dataset[idx][100:200, 100:200], cmap="gray", vmin=-3, vmax=3)
    plt.axis("off")

    plt.show()

    dataset = VanHateren(
        "/data/datasets/vanhateren",
        local_contrast_window=8,
        mode="imc",
        clip=(-3, 3),
        crop_border=32,
    )
    n_images_statistics = 25
    epsilons = np.logspace(-5, 0, 6)
    quantiles = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
    statistics = np.zeros((len(epsilons), len(quantiles) + 2))
    for i, eps in enumerate(epsilons):
        dataset.zscore_epsilon = eps
        px_values = []
        for j in range(n_images_statistics):
            img = dataset[j]
            px_values.append(img.flatten()[::10])
            statistics[i, -2] = min(statistics[i, -2], np.min(img.flatten()))
            statistics[i, -1] = max(statistics[i, -1], np.max(img.flatten()))
        px_values = np.concatenate(px_values)
        statistics[i, :-2] = np.quantile(px_values, quantiles)

    plt.figure()
    plt.plot(epsilons, statistics, marker=".")
    plt.xscale("log")
    plt.xlabel("epsilon")
    plt.ylabel("px value")
    plt.legend([rf"{q} quantile" for q in quantiles] + ["min", "max"])
    plt.show()
