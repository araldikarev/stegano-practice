from algorithms.lsb import LSB
from algorithms.pm1 import PM1
from algorithms.qim import QIM
from algorithms.pvd import PVD
from app.tui_app import SteganoTuiApp


def main():
    algorithms = [
        LSB(),
        PM1(),
        QIM(),
        PVD()
    ]

    app = SteganoTuiApp(algorithms=algorithms)
    app.run()


if __name__ == "__main__":
    main()