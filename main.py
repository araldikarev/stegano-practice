from algorithms.lsb import LSB
from algorithms.pm1 import PM1
from algorithms.qim import QIM
from algorithms.pvd import PVD
from algorithms.dct_interblock_watermark import DCTInterBlockWatermark
from app.tui_app import SteganoTuiApp


def main():
    text_algorithms = [
        LSB(),
        PM1(),
        QIM(),
        PVD(),
    ]

    watermark_algorithms = [
        DCTInterBlockWatermark(),
    ]

    app = SteganoTuiApp(
        text_algorithms=text_algorithms,
        watermark_algorithms=watermark_algorithms,
    )
    app.run()


if __name__ == "__main__":
    main()
