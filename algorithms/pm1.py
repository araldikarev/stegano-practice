from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image
import random

from algorithms.stegano_text_base import SteganoTextBase


class PM1(SteganoTextBase):
    def get_name(self) -> str:
        return "PM1 (Blue channel)"

    def get_description(self) -> str:
        return "PM1: 4 байта длины + UTF-8 payload в младших битах Blue канала."

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

        if len(reshaped_blue) < len(msg_bits):
            raise ValueError("Слишком длинное сообщение для данного изображения")
        for i in range(len(msg_bits)):
            if 0 < reshaped_blue[i] < 255:
                value = (reshaped_blue[i] % 2 + msg_bits[i]) % 2
                reshaped_blue[i] = (
                    reshaped_blue[i] + value
                    if random.choice([-1, 1]) == 1
                    else reshaped_blue[i] - value
                )
            elif reshaped_blue[i] == 0:
                reshaped_blue[i] += 1 if reshaped_blue[i] % 2 != msg_bits[i] else 0
            elif reshaped_blue[i] == 255:
                reshaped_blue[i] -= 1 if reshaped_blue[i] % 2 != msg_bits[i] else 0

        new_blue_channel = reshaped_blue.reshape(blue_channel.shape)
        image_arr[:, :, 2] = new_blue_channel

        stego_image = self.array_to_image(image_arr.astype(np.uint8), mode="RGB")

        return stego_image, {"bits_embedded": len(msg_bits)}

    def extract(self, stego_image: Image.Image, **kwargs: Any) -> str:

        stego_arr = self.image_to_array(stego_image, mode="RGB")
        blue_channel = stego_arr[:, :, 2]
        reshaped_blue = blue_channel.reshape(-1)

        if len(reshaped_blue) < 32:
            raise ValueError(
                "Изображение слишком маленькое, невозможно прочитать заголовок"
            )
        header_bits = reshaped_blue[:32] % 2
        msg_length = int.from_bytes(self.bits_to_bytes(header_bits), byteorder="big")

        if len(reshaped_blue) < 32 + msg_length * 8:
            raise ValueError(
                "Изображение слишком маленькое, невозможно прочитать содержимое"
            )
        result_bits = reshaped_blue[32 : 32 + msg_length * 8] % 2

        result_bytes = self.bits_to_bytes(result_bits)
        result_msg = result_bytes.decode("utf-8")

        return result_msg

    # endregion
