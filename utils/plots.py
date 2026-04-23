from __future__ import annotations

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def show_histograms(cover: Image.Image, stego: Image.Image) -> None:
    cover_l = np.asarray(cover.convert("L"), dtype=np.uint8).ravel()
    stego_l = np.asarray(stego.convert("L"), dtype=np.uint8).ravel()

    plt.figure("Histograms (brightness)")
    plt.hist(cover_l, bins=256, range=(0, 255), alpha=0.5, label="cover")
    plt.hist(stego_l, bins=256, range=(0, 255), alpha=0.5, label="stego")
    plt.title("Brightness histogram: cover vs stego")
    plt.xlabel("Intensity")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.show()