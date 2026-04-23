from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image

from algorithms.stegano_base import SteganoBase


@dataclass
class MetricsPack:
    mse: float | None = None
    psnr: float | None = None
    rmse: float | None = None
    ssim: float | None = None
    ec_bpp: float | None = None
    ber: float | None = None
    ncc: float | None = None


def compute_metrics_pack(
    cover: Image.Image,
    stego: Image.Image,
    *,
    bits_embedded: int | None = None,
    original_message: str | None = None,
    extracted_message: str | None = None,
) -> MetricsPack:
    pack = MetricsPack()

    pack.mse = mse(cover, stego)
    pack.psnr = psnr_from_mse(pack.mse)
    pack.rmse = rmse_from_mse(pack.mse)
    pack.ssim = ssim(cover, stego)
    pack.ncc = ncc(cover, stego)

    if bits_embedded is not None:
        pack.ec_bpp = ec_bpp(bits_embedded, cover)

    if (
        original_message is not None
        and extracted_message is not None
        and bits_embedded is not None
    ):
        pack.ber = ber_from_text(original_message, extracted_message, bits_embedded)

    return pack


def mse(cover: Image.Image, stego: Image.Image) -> float:
    a = np.asarray(cover.convert("RGB"), dtype=np.float32)
    b = np.asarray(stego.convert("RGB"), dtype=np.float32)
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")
    return float(np.mean((a - b) ** 2))


def psnr_from_mse(mse_value: float) -> float:
    if mse_value == 0:
        return float("inf")
    return float(10.0 * np.log10((255.0**2) / mse_value))


def rmse_from_mse(mse_value: float) -> float:
    return float(np.sqrt(mse_value))


def ssim(cover: Image.Image, stego: Image.Image) -> float:
    cover_flat = np.asarray(cover.convert("RGB"), dtype=np.float64).flatten()
    stego_flat = np.asarray(stego.convert("RGB"), dtype=np.float64).flatten()

    cover_mean = np.mean(cover_flat)
    stego_mean = np.mean(stego_flat)

    cover_variance = np.var(cover_flat)
    stego_variance = np.var(stego_flat)

    covariance_matrix = np.cov(cover_flat, stego_flat)
    pixel_covariance = covariance_matrix[0, 1]

    dynamic_range = 255
    k1, k2 = 0.01, 0.03
    luminance_stabilizer = (k1 * dynamic_range) ** 2  # C1
    contrast_stabilizer = (k2 * dynamic_range) ** 2  # C2

    numerator = (2 * cover_mean * stego_mean + luminance_stabilizer) * (
        2 * pixel_covariance + contrast_stabilizer
    )

    denominator = (cover_mean**2 + stego_mean**2 + luminance_stabilizer) * (
        cover_variance + stego_variance + contrast_stabilizer
    )

    return float(numerator/denominator)


def ec_bpp(bits_embedded: int, cover: Image.Image) -> float:
    w, h = cover.size
    return float(bits_embedded) / float(w * h)


def ber_from_text(original: str, extracted: str, bits_embedded: int) -> float:
    original_bits = SteganoBase.bytes_to_bits(original.encode("utf-8"))
    extracted_bits = SteganoBase.bytes_to_bits(extracted.encode("utf-8"))
    return np.sum(original_bits != extracted_bits) / bits_embedded


def ncc(cover: Image.Image, stego: Image.Image) -> float:
    cover_flat = np.asarray(cover.convert("RGB"), dtype=np.float64).flatten()
    stego_flat = np.asarray(stego.convert("RGB"), dtype=np.float64).flatten()
    numerator = np.sum(cover_flat * stego_flat)
    denominator = np.sqrt(np.sum(cover_flat**2) * np.sum(stego_flat**2))

    if denominator == 0:
        return 0.0

    return float(numerator / denominator)
