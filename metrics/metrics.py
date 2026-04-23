from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image


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

    if bits_embedded is not None:
        pack.ec_bpp = ec_bpp(bits_embedded, cover)

    if original_message is not None and extracted_message is not None:
        pack.ber = ber_from_text(original_message, extracted_message)

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
    raise NotImplementedError("SSIM: реализуй по методичке (обязательная метрика).")


def ec_bpp(bits_embedded: int, cover: Image.Image) -> float:
    w, h = cover.size
    return float(bits_embedded) / float(w * h)


def ber_from_text(original: str, extracted: str) -> float:
    raise NotImplementedError("BER: лучше считать по битам payload (или хотя бы по bytes).")


def ncc(a: Any, b: Any) -> float:
    raise NotImplementedError("NCC: реализуй под свою задачу (текст/картинка/вектор).")