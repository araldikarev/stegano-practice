from __future__ import annotations

from pathlib import Path

from PIL import Image

from algorithms.pvd import PVD
from metrics.metrics import compute_metrics_pack, mse, psnr_from_mse, ssim
from robustness.common import (
    ASSETS_DIR,
    RESULTS_DIR,
    ensure_test_image,
    jpeg_roundtrip,
    random_ascii_message,
)


def run(image_path: Path | None = None) -> None:
    image_path = image_path or (ASSETS_DIR / "test_image.jpeg")
    image_path = ensure_test_image(image_path)

    algo = PVD()
    cover = Image.open(image_path)
    cover.load()

    cap_bits = int(algo.capacity_bits(cover) or 0)
    msg_bytes = max(1, min(4096, cap_bits // 16 - 4))
    message = random_ascii_message(msg_bytes, seed=1)

    stego, extra = algo.embed(cover, message)
    bits_embedded = int(extra.get("bits_embedded", 0)) if isinstance(extra, dict) else 0

    base_extracted = algo.extract(stego)
    base = compute_metrics_pack(
        cover,
        stego,
        bits_embedded=bits_embedded,
        original_message=message,
        extracted_message=base_extracted,
    )

    print(f"[PVD][BASE] bits={bits_embedded} PSNR={base.psnr} SSIM={base.ssim} BER={base.ber} NCC={base.ncc}")

    for q in [95, 85, 75, 60, 50, 35]:
        attacked = jpeg_roundtrip(stego, quality=q)
        out_path = RESULTS_DIR / f"pvd_jpeg_q{q}.png"
        attacked.save(out_path)

        ok = True
        err = None
        try:
            extracted = algo.extract(attacked)
        except Exception as ex:
            ok = False
            err = str(ex)
            extracted = ""

        pack = compute_metrics_pack(
            cover,
            attacked,
            bits_embedded=bits_embedded,
            original_message=message,
            extracted_message=extracted,
        )

        attack_psnr = psnr_from_mse(mse(stego, attacked))
        attack_ssim = ssim(stego, attacked)

        print(
            f"[PVD][JPEG q={q}] ok={ok} "
            f"BER={pack.ber} NCC={pack.ncc} "
            f"PSNR(cover)={pack.psnr} SSIM(cover)={pack.ssim} "
            f"PSNR(attack)={attack_psnr} SSIM(attack)={attack_ssim} "
            f"err={err}"
        )


if __name__ == "__main__":
    run()