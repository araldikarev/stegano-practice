from __future__ import annotations

from robustness.dct_brightness_test import run as run_brightness
from robustness.dct_jpeg_test import run as run_jpeg
from robustness.dct_noise_test import run as run_noise


def main() -> None:
    run_jpeg()
    print()
    run_brightness()
    print()
    run_noise()


if __name__ == "__main__":
    main()
