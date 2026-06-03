from __future__ import annotations

from pathlib import Path

from PIL import Image

from algorithms.dct_interblock_watermark import DCTInterBlockWatermark
from metrics.metrics import compute_metrics_pack, mse, psnr_from_mse, ssim
from robustness.common import (
    ASSETS_DIR,
    RESULTS_DIR,
    build_test_watermark,
    ensure_test_image,
    jpeg_roundtrip,
)


DEFAULT_KWARGS = {
    "direction": "LR",
    "coeff_row_1_based": 2,
    "coeff_col_1_based": 2,
    "T": 80.0,
    "K": 12.0,
    "Z": 2.0,
    "arnold_iterations": 0,
}


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

    for q in [95, 85, 75, 60, 50, 35]:
        attacked = jpeg_roundtrip(stego, quality=q)
        out_path = RESULTS_DIR / f"dct_jpeg_q{q}.png"
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
            f"[DCT][JPEG q={q}] ok={ok} "
            f"BER={pack.ber} NCC={pack.ncc} "
            f"PSNR(cover)={pack.psnr} SSIM(cover)={pack.ssim} "
            f"PSNR(attack)={attack_psnr} SSIM(attack)={attack_ssim} "
            f"err={err}"
        )


if __name__ == "__main__":
    run()
