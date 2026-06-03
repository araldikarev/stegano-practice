from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable

import numpy as np
from PIL import Image


class SteganoBase(ABC):

    # region Методы реализации
    @abstractmethod
    def get_name(self) -> str:
        pass

    def get_description(self) -> str:
        return ""

    def get_arguments_to_setup(self) -> dict[str, Callable[[str], Any]]:
        return {}

    def capacity_bits(self, cover_image: Image.Image, **kwargs: Any) -> int | None:
        return None

    # endregion

    # region Utils
    @staticmethod
    def image_to_array(img: Image.Image, *, mode: str = "RGB") -> np.ndarray:
        return np.asarray(img.convert(mode), dtype=np.uint8)

    @staticmethod
    def array_to_image(arr: np.ndarray, *, mode: str = "RGB") -> Image.Image:
        arr_u8 = np.clip(arr, 0, 255).astype(np.uint8)
        return Image.fromarray(arr_u8, mode=mode)

    @staticmethod
    def bytes_to_bits(data: bytes) -> np.ndarray:
        return np.unpackbits(np.frombuffer(data, dtype=np.uint8))

    @staticmethod
    def bits_to_bytes(bits: np.ndarray) -> bytes:
        bits_u8 = np.asarray(bits, dtype=np.uint8)
        packed = np.packbits(bits_u8)
        return packed.tobytes()

    # endregion
