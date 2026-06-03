from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


ASSETS_DIR = Path("assets")
RESULTS_DIR = Path("results") / "robustness"


@dataclass
class CaseResult:
    name: str
    ok: bool
    error: str | None
    ber: float | None
    ncc: float | None
    psnr_cover: float | None
    ssim_cover: float | None
    psnr_attack: float | None
    ssim_attack: float | None
    out_path: Path | None


def ensure_dirs() -> None:
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def ensure_test_image(path: Path, *, size: tuple[int, int] = (768, 512)) -> Path:
    ensure_dirs()
    if path.exists():
        return path

    w, h = size
    x = np.linspace(0, 255, w, dtype=np.uint8)
    y = np.linspace(0, 255, h, dtype=np.uint8)
    xv, yv = np.meshgrid(x, y)

    r = xv
    g = yv
    b = ((xv.astype(np.uint16) + yv.astype(np.uint16)) // 2).astype(np.uint8)
    arr = np.stack([r, g, b], axis=2)

    img = Image.fromarray(arr, mode="RGB")
    dr = ImageDraw.Draw(img)
    dr.rectangle([20, 20, w - 20, h - 20], outline=(255, 255, 255), width=3)
    dr.ellipse([w // 3, h // 4, w // 3 + 140, h // 4 + 140], outline=(0, 0, 0), width=5)
    dr.text((30, h - 40), "PVD robustness test image", fill=(255, 255, 0))

    img.save(path, format="JPEG", quality=95, subsampling=0)
    return path


def random_ascii_message(max_bytes: int, *, seed: int = 123) -> str:
    rng = np.random.default_rng(seed)
    alphabet = np.frombuffer(
        b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,:;!?()[]{}+-=*/_",
        dtype=np.uint8,
    )
    if max_bytes <= 0:
        return ""
    picked = rng.choice(alphabet, size=int(max_bytes), replace=True).astype(np.uint8).tobytes()
    return picked.decode("ascii")


def jpeg_roundtrip(img: Image.Image, *, quality: int) -> Image.Image:
    bio = BytesIO()
    img.convert("RGB").save(bio, format="JPEG", quality=int(quality), optimize=True)
    bio.seek(0)
    out = Image.open(bio)
    out.load()
    return out.convert("RGB")


def build_test_watermark(size: tuple[int, int]) -> Image.Image:
    width, height = size
    if width <= 0 or height <= 0:
        raise ValueError(f"Некорректный размер ЦВЗ: {size}")

    arr = np.zeros((height, width), dtype=np.uint8)
    yy, xx = np.indices((height, width))
    arr[(xx + yy) % 2 == 0] = 255

    inner_top = max(1, height // 6)
    inner_bottom = max(inner_top + 1, height - height // 6)
    inner_left = max(1, width // 6)
    inner_right = max(inner_left + 1, width - width // 6)
    arr[inner_top:inner_bottom, inner_left:inner_right] = 0

    diag_limit = min(height, width)
    arr[np.arange(diag_limit), np.arange(diag_limit)] = 255
    arr[np.arange(diag_limit), width - 1 - np.arange(diag_limit)] = 255

    return Image.fromarray(arr, mode="L")
