from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from algorithms.dct_interblock_watermark import DCTInterBlockWatermark
from metrics.metrics import compute_metrics_pack, mse, psnr_from_mse, ssim
from robustness.common import ASSETS_DIR, RESULTS_DIR, build_test_watermark, ensure_test_image


DEFAULT_KWARGS = {
    "direction": "LR",
    "coeff_row_1_based": 4,
    "coeff_col_1_based": 4,
    "T": 80.0,
    "K": 12.0,
    "Z": 2.0,
    "arnold_iterations": 0,
}


def add_gaussian_noise_rgb(img: Image.Image, *, sigma: float, seed: int) -> Image.Image:
    rng = np.random.default_rng(seed)
    arr = np.asarray(img.convert("RGB"), dtype=np.float32)
    noise = rng.normal(0.0, float(sigma), size=arr.shape).astype(np.float32)
    out = np.clip(arr + noise, 0.0, 255.0).astype(np.uint8)
    return Image.fromarray(out, mode="RGB")


def _watermark_size(cover: Image.Image, direction: str) -> tuple[int, int]:
    blocks_x = cover.width // 8
    blocks_y = cover.height // 8
    if direction in {"LR", "RL"}:
        return max(0, blocks_x - 1), blocks_y
    return blocks_x, max(0, blocks_y - 1)


def run(image_path: Path | None = None) -> None:
    image_path = image_path or (ASSETS_DIR / "test_image.jpeg")
    image_path = ensure_test_image(image_path)

    algo = DCTInterBlockWatermark()
    cover = Image.open(image_path)
    cover.load()

    watermark = build_test_watermark(_watermark_size(cover, DEFAULT_KWARGS["direction"]))
    stego, extra = algo.embed(cover, watermark, **DEFAULT_KWARGS)
    bits_embedded = int(extra.get("bits_embedded", 0)) if isinstance(extra, dict) else 0
    prepared_watermark = extra["prepared_watermark"]

    base_extracted = algo.extract(stego, **DEFAULT_KWARGS)
    base = compute_metrics_pack(
        cover,
        stego,
        bits_embedded=bits_embedded,
        original_watermark=prepared_watermark,
        extracted_watermark=base_extracted,
    )
    print(f"[DCT][BASE] bits={bits_embedded} PSNR={base.psnr} SSIM={base.ssim} BER={base.ber} NCC={base.ncc}")

    for sigma in [0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0]:
        attacked = add_gaussian_noise_rgb(stego, sigma=sigma, seed=10 + int(sigma * 10))
        out_path = RESULTS_DIR / f"dct_noise_sigma{sigma:.1f}.png"
        attacked.save(out_path)

        ok = True
        err = None
        try:
            extracted = algo.extract(attacked, **DEFAULT_KWARGS)
        except Exception as ex:
            ok = False
            err = str(ex)
            extracted = Image.new("L", prepared_watermark.size, color=0)

        pack = compute_metrics_pack(
            cover,
            attacked,
            bits_embedded=bits_embedded,
            original_watermark=prepared_watermark,
            extracted_watermark=extracted,
        )

        attack_psnr = psnr_from_mse(mse(stego, attacked))
        attack_ssim = ssim(stego, attacked)

        print(
            f"[DCT][NOISE σ={sigma:.1f}] ok={ok} "
            f"BER={pack.ber} NCC={pack.ncc} "
            f"PSNR(cover)={pack.psnr} SSIM(cover)={pack.ssim} "
            f"PSNR(attack)={attack_psnr} SSIM(attack)={attack_ssim} "
            f"err={err}"
        )


if __name__ == "__main__":
    run()
