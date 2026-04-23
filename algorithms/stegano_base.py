from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Union

import numpy as np
from PIL import Image

PathLike = Union[str, Path]


class SteganoBase(ABC):

    # region Методы реализации
    @abstractmethod
    def get_name(self) -> str:
        pass

    def get_description(self) -> str:
        return ""

    def get_arguments_to_setup(self) -> dict[str, Callable[[str], Any]]:
        return {}

    @abstractmethod
    def embed(
        self,
        cover_image: Image.Image,
        message: str,
        **kwargs: Any,
    ) -> tuple[Image.Image, dict[str, Any]]:
        pass

    @abstractmethod
    def extract(
        self,
        stego_image: Image.Image,
        **kwargs: Any,
    ) -> str:
        pass

    # endregion

    # region Utils

    def embed_path(
        self,
        cover_path: PathLike,
        message: str,
        **kwargs: Any,
    ) -> tuple[Image.Image, dict[str, Any]]:
        cover = self.load_image(cover_path)
        return self.embed(cover, message, **kwargs)

    def extract_path(self, stego_path: PathLike, **kwargs: Any) -> str:
        stego = self.load_image(stego_path)
        return self.extract(stego, **kwargs)

    @staticmethod
    def load_image(path: PathLike) -> Image.Image:
        img = Image.open(path)
        img.load()
        return img

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
