from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image

from algorithms.stegano_base import SteganoBase


class QIM(SteganoBase):
    def get_name(self) -> str:
        return "QIM (Blue Channel)"

    def get_description(self) -> str:
        return "QIM: 4 байта длины + UTF-8 payload в младших битах Blue канала."

    def get_arguments_to_setup(self):
        return {
            "q": self.validate_q,  # Шаг квантования
        }

    # region Валидация
    def validate_q(self, value: float):
        try:
            f_value = float(value)
        except Exception as ex:
            raise ValueError(f"Ошибка: q должно быть float'ом: {ex}")
        if f_value < 1:
            raise ValueError("Ошибка: q квантование не может быть меньше 1")
        return f_value

    # endregion

    # region Реализация алгоритма
    def embed(
        self,
        cover_image: Image.Image,
        message: str,
        **kwargs: Any,
    ) -> tuple[Image.Image, dict[str, Any]]:

        image_arr = self.image_to_array(cover_image, mode="RGB").copy()
        blue_channel = image_arr[:, :, 2]
        reshaped_blue = blue_channel.reshape(-1)

        msg_bytes = message.encode("utf-8")
        msg_header = len(msg_bytes).to_bytes(4, byteorder="big")
        msg_payload = msg_header + msg_bytes
        msg_bits = self.bytes_to_bits(msg_payload)

        q = float(kwargs.get("q", 10.0))

        if len(reshaped_blue) < len(msg_bits):
            raise ValueError("Слишком длинное сообщение для данного изображения")

        for i in range(len(msg_bits)):
            shift = msg_bits[i] * q / 2
            value = float(reshaped_blue[i])
            quantized = round((value - shift) / q) * q + shift
            reshaped_blue[i] = np.clip(round(quantized), 0, 255)

        new_blue_channel = reshaped_blue.reshape(blue_channel.shape)
        image_arr[:, :, 2] = new_blue_channel

        stego_image = self.array_to_image(image_arr.astype(np.uint8), mode="RGB")

        return stego_image, {"bits_embedded": len(msg_bits)}

    def extract(self, stego_image: Image.Image, **kwargs: Any) -> str:

        stego_arr = self.image_to_array(stego_image, mode="RGB")
        blue_channel = stego_arr[:, :, 2]
        reshaped_blue = blue_channel.reshape(-1)

        q = float(kwargs.get("q", 10.0))

        if len(reshaped_blue) < 32:
            raise ValueError(
                "Изображение слишком маленькое, невозможно прочитать заголовок"
            )

        header_bits = self._lattice_compare(reshaped_blue[:32], q)
        msg_length = int.from_bytes(self.bits_to_bytes(header_bits), byteorder="big")

        if len(reshaped_blue) < 32 + msg_length * 8:
            raise ValueError(
                "Изображение слишком маленькое, невозможно прочитать содержимое"
            )
        result_bits = self._lattice_compare(reshaped_blue[32 : 32 + msg_length * 8], q)

        result_bytes = self.bits_to_bytes(result_bits)
        result_msg = result_bytes.decode("utf-8")

        return result_msg

    # endregion

    # region вспомогательные функции

    def _lattice_compare(self, pixels: np.ndarray, q: float) -> np.ndarray:
        pixels_f = pixels.astype(float)

        target0 = np.round(pixels_f / q) * q
        error0 = np.abs(pixels_f - target0)

        shift = q / 2
        target1 = np.round((pixels_f - shift) / q) * q + shift
        error1 = np.abs(pixels_f - target1)

        return (error0 > error1).astype(np.uint8)

    # endregion
