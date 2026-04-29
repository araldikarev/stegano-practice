from __future__ import annotations

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def _rgb(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("RGB"), dtype=np.uint8)


def _luma_flat(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("L"), dtype=np.uint8).ravel()


def _channel_flat(img: Image.Image, channel: int) -> np.ndarray:
    return _rgb(img)[:, :, channel].ravel()


def _abs_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.abs(a.astype(np.int16) - b.astype(np.int16)).astype(np.uint8)


def show_hist_brightness(cover: Image.Image, stego: Image.Image) -> None:
    cover_l = _luma_flat(cover)
    stego_l = _luma_flat(stego)

    plt.figure("Histogram: Brightness (L)")
    plt.hist(cover_l, bins=256, range=(0, 255), alpha=0.5, label="cover")
    plt.hist(stego_l, bins=256, range=(0, 255), alpha=0.5, label="stego")
    plt.title("Brightness histogram (L): cover vs stego")
    plt.xlabel("Intensity")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.show()


def show_hist_channel(cover: Image.Image, stego: Image.Image, channel: int, name: str) -> None:
    c = _channel_flat(cover, channel)
    s = _channel_flat(stego, channel)

    plt.figure(f"Histogram: {name} channel")
    plt.hist(c, bins=256, range=(0, 255), alpha=0.5, label="cover")
    plt.hist(s, bins=256, range=(0, 255), alpha=0.5, label="stego")
    plt.title(f"{name} channel histogram: cover vs stego")
    plt.xlabel("Intensity")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.show()


def show_hist_rgb(cover: Image.Image, stego: Image.Image) -> None:
    c = _rgb(cover)
    s = _rgb(stego)

    fig, axes = plt.subplots(3, 1, num="Histograms: RGB channels", figsize=(9, 10), sharex=True)
    channels = [(0, "Red"), (1, "Green"), (2, "Blue")]

    for ax, (idx, name) in zip(axes, channels):
        ax.hist(c[:, :, idx].ravel(), bins=256, range=(0, 255), alpha=0.5, label="cover")
        ax.hist(s[:, :, idx].ravel(), bins=256, range=(0, 255), alpha=0.5, label="stego")
        ax.set_title(f"{name} channel")
        ax.set_ylabel("Count")
        ax.legend()

    axes[-1].set_xlabel("Intensity")
    plt.tight_layout()
    plt.show()


def show_abs_diff_hist_brightness(cover: Image.Image, stego: Image.Image) -> None:
    d = _abs_diff(np.asarray(cover.convert("L"), dtype=np.uint8),
                  np.asarray(stego.convert("L"), dtype=np.uint8)).ravel()

    plt.figure("Histogram: |L cover - L stego|")
    plt.hist(d, bins=256, range=(0, 255), alpha=0.85, label="abs diff")
    plt.title("Abs diff histogram (L)")
    plt.xlabel("|Δ|")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()


def show_abs_diff_hist_channel(cover: Image.Image, stego: Image.Image, channel: int, name: str) -> None:
    c = _rgb(cover)[:, :, channel]
    s = _rgb(stego)[:, :, channel]
    d = _abs_diff(c, s).ravel()

    plt.figure(f"Histogram: |{name} cover - {name} stego|")
    plt.hist(d, bins=256, range=(0, 255), alpha=0.85, label="abs diff")
    plt.title(f"Abs diff histogram ({name} channel)")
    plt.xlabel("|Δ|")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()