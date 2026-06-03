from __future__ import annotations

from abc import abstractmethod
from typing import Any

from PIL import Image

from algorithms.stegano_base import SteganoBase


class SteganoWatermarkBase(SteganoBase):
    """Контракт для методов цифрового водяного знака.

    Payload здесь — изображение-ЦВЗ, а не текстовая строка.
    Конкретная бинаризация/нормализация ЦВЗ относится к реализации алгоритма.
    """

    @abstractmethod
    def embed(
        self,
        cover_image: Image.Image,
        watermark_image: Image.Image,
        **kwargs: Any,
    ) -> tuple[Image.Image, dict[str, Any]]:
        pass

    @abstractmethod
    def extract(
        self,
        stego_image: Image.Image,
        **kwargs: Any,
    ) -> Image.Image:
        pass
