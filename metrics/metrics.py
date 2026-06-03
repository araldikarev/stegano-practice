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
    original_watermark: Image.Image | np.ndarray | None = None,
    extracted_watermark: Image.Image | np.ndarray | None = None,
) -> MetricsPack:
    pack = MetricsPack()

    pack.mse = mse(cover, stego)
    pack.psnr = psnr_from_mse(pack.mse)
    pack.rmse = rmse_from_mse(pack.mse)
    pack.ssim = ssim(cover, stego)
    if bits_embedded is not None:
        pack.ec_bpp = ec_bpp(bits_embedded, cover)

    if original_message is not None and extracted_message is not None:
        pack.ncc = ncc(original_message, extracted_message, bits_limit=bits_embedded)
        pack.ber = ber_from_text(original_message, extracted_message, bits_embedded)
    elif original_watermark is not None and extracted_watermark is not None:
        pack.ncc = ncc_from_watermarks(
            original_watermark,
            extracted_watermark,
            bits_limit=bits_embedded,
        )
        pack.ber = ber_from_watermarks(
            original_watermark,
            extracted_watermark,
            bits_limit=bits_embedded,
        )
    else:
        pack.ber = None
        pack.ncc = None

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


def ber_from_bit_vectors(
    original_bits: np.ndarray,
    extracted_bits: np.ndarray,
    bits_limit: int | None = None,
) -> float:
    total_bits = int(original_bits.size)
    if total_bits == 0:
        return 0.0

    if bits_limit is not None:
        total_bits = min(total_bits, int(bits_limit))

    original_view = original_bits[:total_bits]
    extracted_view = extracted_bits[:total_bits]
    compared_bits = min(total_bits, int(extracted_view.size))

    errors = int(np.sum(original_view[:compared_bits] != extracted_view[:compared_bits]))
    errors += total_bits - compared_bits
    return float(errors) / float(total_bits)


def ber_from_text(original: str, extracted: str, bits_embedded: int) -> float:
    original_bits = SteganoBase.bytes_to_bits(original.encode("utf-8"))
    extracted_bits = SteganoBase.bytes_to_bits(extracted.encode("utf-8"))
    return ber_from_bit_vectors(original_bits, extracted_bits, bits_embedded)


def ncc_from_bit_vectors(
    original_bits: np.ndarray,
    extracted_bits: np.ndarray,
    bits_limit: int | None = None,
) -> float:
    original_view = original_bits.astype(np.float64)
    extracted_view = extracted_bits.astype(np.float64)

    if original_view.size == 0:
        return 0.0

    total_bits = int(original_view.size)
    if bits_limit is not None:
        total_bits = min(total_bits, int(bits_limit))

    original_view = original_view[:total_bits]
    if extracted_view.size < total_bits:
        extracted_view = np.pad(
            extracted_view,
            (0, total_bits - int(extracted_view.size)),
            constant_values=0.0,
        )
    else:
        extracted_view = extracted_view[:total_bits]

    numerator = float(np.sum(original_view * extracted_view))
    denominator = float(
        np.sqrt(np.sum(original_view * original_view) * np.sum(extracted_view * extracted_view))
    )

    if denominator == 0.0:
        return 0.0
    return numerator / denominator


def ncc(original: str, extracted: str, bits_limit: int | None = None) -> float:
    original_bits = SteganoBase.bytes_to_bits(original.encode("utf-8"))
    extracted_bits = SteganoBase.bytes_to_bits(extracted.encode("utf-8"))
    return ncc_from_bit_vectors(original_bits, extracted_bits, bits_limit)


def _watermark_to_bits(watermark: Image.Image | np.ndarray) -> np.ndarray:
    if isinstance(watermark, Image.Image):
        arr = np.asarray(watermark.convert("L"), dtype=np.uint8)
    else:
        arr = np.asarray(watermark)
        if arr.ndim == 3:
            arr = arr[..., 0]
        arr = arr.astype(np.uint8)
    return (arr > 127).astype(np.uint8).reshape(-1)


def ber_from_watermarks(
    original: Image.Image | np.ndarray,
    extracted: Image.Image | np.ndarray,
    bits_limit: int | None = None,
) -> float:
    return ber_from_bit_vectors(
        _watermark_to_bits(original),
        _watermark_to_bits(extracted),
        bits_limit,
    )


def ncc_from_watermarks(
    original: Image.Image | np.ndarray,
    extracted: Image.Image | np.ndarray,
    bits_limit: int | None = None,
) -> float:
    return ncc_from_bit_vectors(
        _watermark_to_bits(original),
        _watermark_to_bits(extracted),
        bits_limit,
    )
