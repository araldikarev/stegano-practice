from __future__ import annotations

from abc import abstractmethod
from typing import Any

from PIL import Image

from algorithms.stegano_base import SteganoBase


class SteganoTextBase(SteganoBase):
    """Контракт для методов скрытия текстового сообщения."""

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
