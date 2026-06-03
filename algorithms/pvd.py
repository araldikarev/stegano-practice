from __future__ import annotations

import math
from typing import Any

import numpy as np
from PIL import Image

from algorithms.stegano_text_base import SteganoTextBase


class PVD(SteganoTextBase):

    RANGES = [(0, 7), (8, 15), (16, 31), (32, 63), (64, 127), (128, 255)]

    def get_name(self) -> str:
        return "PVD (Blue Channel)"

    def get_description(self) -> str:
        return "PVD: 4 байта длины + UTF-8 payload в битах Blue канала."
    
    def capacity_bits(self, cover_image: Image.Image, **kwargs: Any) -> int:
        image_arr = self.image_to_array(cover_image, mode="RGB")
        blue_channel = image_arr[:, :, 2]
        reshaped_blue = blue_channel.reshape(-1)

        capacity = 0
        for i in range(0, len(reshaped_blue) - 1, 2):
            p0 = int(reshaped_blue[i])
            p1 = int(reshaped_blue[i + 1])
            abs_dk = abs(p0 - p1)
            found_range = self._find_range(abs_dk)
            if found_range is None:
                continue
            capacity += self._n_for(found_range)
        return int(capacity)

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
        msg_size = int(msg_bits.size)

        capacity = int(self.capacity_bits(cover_image, **kwargs) or 0)
        if capacity < msg_size:
            raise ValueError(
                f"Слишком длинное сообщение: capacity={capacity} бит, нужно={msg_size} бит."
            )

        bit_index = 0
        for i in range(0, len(reshaped_blue) - 1, 2):

            p0 = int(reshaped_blue[i])
            p1 = int(reshaped_blue[i + 1])

            dk = p0 - p1
            abs_dk = abs(dk)
            found_range = self._find_range(abs_dk)
            if found_range is None:
                continue

            n = self._n_for(found_range)
            if n <= 0:
                continue

            current_bits = msg_bits[bit_index : bit_index + n]
            if len(current_bits) < n:
                current_bits = np.pad(
                    current_bits, (0, n - len(current_bits)), constant_values=0
                )

            shift_value = int("".join(map(str, current_bits.tolist())), 2)
            range_bound = found_range[0]

            dk_star = (range_bound + shift_value) if dk >= 0 else -(range_bound + shift_value)
            step = dk_star - dk

            if dk % 2 != 0:
                p0 += math.ceil(step / 2)
                p1 -= math.floor(step / 2)
            else:
                p0 += math.floor(step / 2)
                p1 -= math.ceil(step / 2)

            if p0 < 0:
                p1 += -p0
                p0 = 0
            if p0 > 255:
                p1 -= p0 - 255
                p0 = 255
            if p1 < 0:
                p0 += -p1
                p1 = 0
            if p1 > 255:
                p0 -= p1 - 255
                p1 = 255

            reshaped_blue[i] = np.clip(p0, 0, 255)
            reshaped_blue[i + 1] = np.clip(p1, 0, 255)

            bit_index += n
            if bit_index >= len(msg_bits):
                break

        new_blue_channel = reshaped_blue.reshape(blue_channel.shape)
        image_arr[:, :, 2] = new_blue_channel

        stego_image = self.array_to_image(image_arr.astype(np.uint8), mode="RGB")

        return stego_image, {"bits_embedded": min(bit_index, msg_size)}

    def extract(self, stego_image: Image.Image, **kwargs: Any) -> str:

        stego_arr = self.image_to_array(stego_image, mode="RGB")
        blue_channel = stego_arr[:, :, 2]
        reshaped_blue = blue_channel.reshape(-1)

        if len(reshaped_blue) < 32:
            raise ValueError(
                "Изображение слишком маленькое, невозможно прочитать заголовок"
            )

        found_bits = []
        bits_count: int | None = None

        for i in range(0, len(reshaped_blue) - 1, 2):
            p0 = int(reshaped_blue[i])
            p1 = int(reshaped_blue[i + 1])
            abs_dk = abs(p0 - p1)


            found_range = self._find_range(abs_dk)
            if found_range is None:
                continue

            n = self._n_for(found_range)
            if n <= 0:
                continue

            value = abs_dk - found_range[0]
            binary = self._to_binary(value, n)
            found_bits.extend(binary)

            if bits_count is None and len(found_bits) >= 32:
                bits_count = (
                    32
                    + int.from_bytes(
                        self.bits_to_bytes(found_bits[:32]), byteorder="big"
                    )
                    * 8
                )
            elif bits_count is not None and len(found_bits) >= bits_count:
                break

        if bits_count is None:
            raise Exception("Не удалось прочитать заголовок")
        if len(found_bits) < bits_count:
            raise Exception("Изображение повреждено.")

        result_bytes = self.bits_to_bytes(found_bits[32:bits_count])
        result_msg = result_bytes.decode("utf-8")

        return result_msg

    # endregion

    # region Вспомогательные функции

    def _to_binary(self, value: int, length: int):
        extracted = []
        for _ in range(length):
            extracted.append(int(value % 2))
            value //= 2
        return extracted[::-1]

    def _find_range(self, distance: int):
        return next((r for r in self.RANGES if r[0] <= distance <= r[1]), None)

    def _n_for(self, find_range: tuple[int, int]) -> int:
        return int(math.log2(find_range[1] - find_range[0] + 1))

    # endregion
