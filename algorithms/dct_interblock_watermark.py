from __future__ import annotations

from typing import Any, Callable

import numpy as np
from scipy.fft import dctn, idctn
from PIL import Image

from algorithms.stegano_watermark_base import SteganoWatermarkBase


class DCTInterBlockWatermark(SteganoWatermarkBase):
    """Метод частотного ЦВЗ на основе межблочной разности ДКП-коэффициентов."""

    DIRECTIONS = {"LR", "UD", "RL", "DU"}

    def get_name(self) -> str:
        return "DCT inter-block (Blue Channel)"

    def get_description(self) -> str:
        return (
            "Частотный метод ЦВЗ на основе межблочной разности "
            "ДКП-коэффициентов синего канала с поддержкой преобразования Арнольда."
        )

    def get_arguments_to_setup(self) -> dict[str, Callable[[str], Any]]:
        return {
            "direction": self._parse_direction,
            "coeff_row_1_based": self._parse_dct_block_coord_1_based,
            "coeff_col_1_based": self._parse_dct_block_coord_1_based,
            "T": self._parse_positive_float,
            "K": self._parse_positive_float,
            "Z": self._parse_positive_float,
            "arnold_iterations": self._parse_non_negative_int,
        }

    def capacity_bits(self, cover_image: Image.Image, **kwargs: Any) -> int | None:
        blocks_y = cover_image.height // 8
        blocks_x = cover_image.width // 8
        direction = str(kwargs.get("direction", "LR")).upper()

        if direction in {"LR", "RL"}:
            return max(0, blocks_y * (blocks_x - 1))
        if direction in {"UD", "DU"}:
            return max(0, (blocks_y - 1) * blocks_x)

        raise ValueError(f"Unsupported direction: {direction}")

    def embed(
        self,
        cover_image: Image.Image,
        watermark_image: Image.Image,
        **kwargs: Any,
    ) -> tuple[Image.Image, dict[str, Any]]:

        cover_rgb = self.image_to_array(cover_image, mode="RGB").copy().astype(np.float64)
        cover_arr = cover_rgb[:, :, 2]

        row = kwargs.get("coeff_row_1_based", 4) - 1
        col = kwargs.get("coeff_col_1_based", 4) - 1
        direction = kwargs.get("direction", "LR")
        arnold_iterations = kwargs.get("arnold_iterations", 0)
        T = kwargs.get("T", 80.0)
        K = kwargs.get("K", 12.0)
        Z = kwargs.get("Z", 2.0)

        blocks_y = cover_image.height // 8
        blocks_x = cover_image.width // 8

        dct_grid = np.empty((blocks_y, blocks_x, 8, 8), dtype=np.float64)

        watermark_l = watermark_image.convert("L").resize(
            (blocks_x, blocks_y),
            Image.Resampling.NEAREST,
        )
        watermark_arr = (np.asarray(watermark_l, dtype=np.uint8) > 127).astype(np.uint8)
        watermark_arnold = self._arnold_transform(
            watermark_arr,
            iterations=arnold_iterations,
        )

        for p in range(blocks_y):
            for q in range(blocks_x):
                block = cover_arr[p * 8 : (p + 1) * 8, q * 8 : (q + 1) * 8]
                block_centered = block - 128.0
                dct_grid[p, q] = dctn(block_centered, norm="ortho")

        if direction == "LR":
            p_range = range(blocks_y)
            q_range = range(blocks_x - 2, -1, -1)
            bits_embedded = blocks_y * max(0, blocks_x - 1)

        elif direction == "RL":
            p_range = range(blocks_y)
            q_range = range(1, blocks_x)
            bits_embedded = blocks_y * max(0, blocks_x - 1)

        elif direction == "UD":
            p_range = range(blocks_y - 2, -1, -1)
            q_range = range(blocks_x)
            bits_embedded = max(0, blocks_y - 1) * blocks_x

        elif direction == "DU":
            p_range = range(1, blocks_y)
            q_range = range(blocks_x)
            bits_embedded = max(0, blocks_y - 1) * blocks_x

        else:
            raise ValueError(f"Unsupported direction: {direction}")

        for p in p_range:
            for q in q_range:
                match direction:
                    case "LR":
                        neighbor_p, neighbor_q = p, q + 1
                    case "UD":
                        neighbor_p, neighbor_q = p + 1, q
                    case "RL":
                        neighbor_p, neighbor_q = p, q - 1
                    case "DU":
                        neighbor_p, neighbor_q = p - 1, q
                    case _:
                        raise ValueError(f"Unsupported direction: {direction}")

                if (
                    neighbor_p < 0
                    or neighbor_p >= blocks_y
                    or neighbor_q < 0
                    or neighbor_q >= blocks_x
                ):
                    continue

                dct_block = dct_grid[p, q]
                neighbor_block = dct_grid[neighbor_p, neighbor_q]

                current_dc = dct_block[0, 0]

                ac_coords = [
                    (0, 1), (1, 0), (1, 1),
                    (0, 2), (2, 0),
                    (2, 1), (1, 2),
                    (0, 3), (3, 0),
                ]
                ac_values = [dct_block[r, c] for r, c in ac_coords]
                median = np.median(ac_values)

                if abs(current_dc) > 1000 or abs(current_dc) < 1:
                    M = abs(Z * median)
                else:
                    M = abs(Z * (current_dc - median) / current_dc)

                if M < 1e-6:
                    M = 1.0

                coeff_to_modify = dct_block[row, col]
                coeff_neighbor = neighbor_block[row, col]
                delta = coeff_to_modify - coeff_neighbor

                bit = watermark_arnold[p, q]

                if bit == 1:
                    if delta > T - K:
                        while delta > T - K:
                            coeff_to_modify -= M
                            delta = coeff_to_modify - coeff_neighbor

                    elif K > delta > -T / 2.0:
                        while delta < K:
                            coeff_to_modify += M
                            delta = coeff_to_modify - coeff_neighbor

                    elif delta < -T / 2.0:
                        while delta > -T - K:
                            coeff_to_modify -= M
                            delta = coeff_to_modify - coeff_neighbor

                else:
                    if delta > T / 2.0:
                        while delta <= T + K:
                            coeff_to_modify += M
                            delta = coeff_to_modify - coeff_neighbor

                    elif -K < delta < T / 2.0:
                        while delta >= -K:
                            coeff_to_modify -= M
                            delta = coeff_to_modify - coeff_neighbor

                    elif delta < K - T:
                        while delta <= K - T:
                            coeff_to_modify += M
                            delta = coeff_to_modify - coeff_neighbor

                dct_block[row, col] = coeff_to_modify

        stego_channel = cover_arr.copy()

        for p in range(blocks_y):
            for q in range(blocks_x):
                block_dct = dct_grid[p, q]
                block_idct = idctn(block_dct, norm="ortho") + 128.0
                stego_channel[p * 8 : (p + 1) * 8, q * 8 : (q + 1) * 8] = block_idct

        stego_channel = np.rint(stego_channel)
        stego_channel = np.clip(stego_channel, 0, 255)

        cover_rgb[:, :, 2] = stego_channel

        stego_rgb_arr = np.clip(cover_rgb, 0, 255).astype(np.uint8)
        stego_image = self.array_to_image(stego_rgb_arr, mode="RGB")

        prepared_wm_image = Image.fromarray((watermark_arr * 255).astype(np.uint8), mode="L")

        return stego_image, {
            "bits_embedded": bits_embedded,
            "prepared_watermark": prepared_wm_image
        }

    def extract(self, stego_image: Image.Image, **kwargs: Any) -> Image.Image:
        row = kwargs.get("coeff_row_1_based", 4) - 1
        col = kwargs.get("coeff_col_1_based", 4) - 1
        direction = kwargs.get("direction", "LR")
        arnold_iterations = kwargs.get("arnold_iterations", 0)
        T = float(kwargs.get("T", 80.0))

        stego_rgb = self.image_to_array(stego_image, mode="RGB").astype(np.float64)
        stego_arr = stego_rgb[:, :, 2]

        height, width = stego_arr.shape
        blocks_y = height // 8
        blocks_x = width // 8

        dct_grid = np.empty((blocks_y, blocks_x, 8, 8), dtype=np.float64)
        for p in range(blocks_y):
            for q in range(blocks_x):
                block = stego_arr[p * 8 : (p + 1) * 8, q * 8 : (q + 1) * 8]
                dct_grid[p, q] = dctn(block - 128.0, norm='ortho')

        extracted_bits = np.ones((blocks_y, blocks_x), dtype=np.uint8)

        for p in range(blocks_y):
            for q in range(blocks_x):
                match direction:
                    case "LR":
                        neighbor_p, neighbor_q = p, q + 1
                    case "UD":
                        neighbor_p, neighbor_q = p + 1, q
                    case "RL":
                        neighbor_p, neighbor_q = p, q - 1
                    case "DU":
                        neighbor_p, neighbor_q = p - 1, q
                    case _:
                        raise ValueError(f"Unsupported direction: {direction}")

                if neighbor_p < 0 or neighbor_p >= blocks_y or neighbor_q < 0 or neighbor_q >= blocks_x:
                    continue

                c = dct_grid[p, q][row, col]
                c_neighbor = dct_grid[neighbor_p, neighbor_q][row, col]
                delta = c - c_neighbor

                if (delta < -T) or (0.0 < delta < T):
                    extracted_bits[p, q] = 1
                else:
                    extracted_bits[p, q] = 0

        extracted_bits_original = self._inverse_arnold_transform(extracted_bits, iterations=arnold_iterations)
        extracted_wm_arr = (extracted_bits_original * 255).astype(np.uint8)
        extracted_wm_image = self.array_to_image(extracted_wm_arr, mode="L")

        return extracted_wm_image

    # region Парсинг параметров
    def _parse_direction(self, value: str) -> str:
        normalized = value.strip().upper()
        if normalized not in self.DIRECTIONS:
            raise ValueError("direction должен быть одним из: LR, UD, RL, DU")
        return normalized

    def _parse_positive_float(self, value: str) -> float:
        try:
            result = float(value)
        except Exception as ex:
            raise ValueError(f"Ожидалось float-число: {ex}")
        if result <= 0:
            raise ValueError("Значение должно быть > 0")
        return result

    def _parse_non_negative_int(self, value: str) -> int:
        try:
            result = int(value)
        except Exception as ex:
            raise ValueError(f"Ожидалось целое число: {ex}")
        if result < 0:
            raise ValueError("Значение должно быть >= 0")
        return result

    def _parse_dct_block_coord_1_based(self, value: str) -> int:
        try:
            result = int(value)
        except Exception as ex:
            raise ValueError(f"Ожидалось целое число от 1 до 8: {ex}")
        if not 1 <= result <= 8:
            raise ValueError("Координата ДКП-блока должна быть в диапазоне 1..8")
        return result
    # endregion

    # region Вспомогательные методы
    def _arnold_transform(self, image: np.ndarray, iterations: int) -> np.ndarray:
        if iterations <= 0:
            return image.copy()
        
        if image.shape[0] != image.shape[1]:
            raise ValueError(
                f"Преобразование Арнольда требует квадратной матрицы."
            )

        current = image.copy()
        for _ in range(iterations):
            zeros = np.zeros_like(image)
            for i in range(image.shape[0]):
                for j in range(image.shape[1]):
                    x_i = (i + j) % image.shape[0]
                    y_j = (i + 2 * j) % image.shape[1]
                    zeros[x_i, y_j] = current[i, j]
            current = zeros

        return current

    def _inverse_arnold_transform(self, image: np.ndarray, iterations: int) -> np.ndarray:
        if iterations <= 0:
            return image.copy()
        
        if image.shape[0] != image.shape[1]:
            raise ValueError(
                f"Обратное преобразование Арнольда требует квадратной матрицы."
            )

        current = image.copy()
        for _ in range(iterations):
            zeros = np.zeros_like(image)
            for i in range(image.shape[0]):
                for j in range(image.shape[1]):
                    x_orig = (2 * i - j) % image.shape[0]
                    y_orig = (-i + j) % image.shape[1]
                    zeros[x_orig, y_orig] = current[i, j]
            current = zeros
        return current
    # endregion