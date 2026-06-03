from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from InquirerPy import inquirer
from PIL import Image

from algorithms.stegano_base import SteganoBase
from algorithms.stegano_text_base import SteganoTextBase
from algorithms.stegano_watermark_base import SteganoWatermarkBase
from utils.cli import ask_parsed, print_error, print_success, pause
from utils.images import list_images, open_in_viewer, unique_out_path
from metrics import compute_metrics_pack
from utils.plots import (
    show_hist_brightness,
    show_hist_channel,
    show_hist_rgb,
    show_abs_diff_hist_brightness,
    show_abs_diff_hist_channel,
)


@dataclass
class TuiConfig:
    images_dir: Path = Path("images")
    results_dir: Path = Path("results")


@dataclass
class AlgorithmChoice:
    kind: Literal["text", "watermark"]
    algorithm: SteganoBase


class SteganoTuiApp:
    def __init__(
        self,
        text_algorithms: list[SteganoTextBase],
        watermark_algorithms: list[SteganoWatermarkBase] | None = None,
        config: TuiConfig | None = None,
    ):
        self.text_algorithms = text_algorithms
        self.watermark_algorithms = watermark_algorithms or []
        self.config = config or TuiConfig()

    def run(self) -> None:
        while True:
            print()
            img_path = self._pick_image()
            if img_path is None:
                print("\nЗавершение.")
                return

            while True:
                print()
                res = self._image_actions(img_path)
                if res == "back":
                    break
                if res == "open":
                    continue
                if res == "choose":
                    next_step = self._work_with_chosen_image(img_path)
                    if next_step == "exit":
                        print("\nЗавершение.")
                        return
                    break

    def _pick_image(self) -> Path | None:
        while True:
            imgs = list_images(self.config.images_dir)
            choices = [p.name for p in imgs] + ["↻ Обновить", "Выход"]

            picked = inquirer.select(
                message=f"Выбор изображения (папка {self.config.images_dir}/):",
                choices=choices,
            ).execute()

            if picked == "Выход":
                return None
            if picked == "↻ Обновить":
                continue
            return self.config.images_dir / picked

    def _image_actions(self, img_path: Path) -> str:
        act = inquirer.select(
            message=f"Файл: {img_path.name}",
            choices=["Открыть (open)", "Выбрать (choose)", "Назад"],
            default="Выбрать (choose)",
        ).execute()

        if act.startswith("Открыть"):
            try:
                open_in_viewer(img_path)
            except Exception as ex:
                print()
                print_error(f"Не удалось открыть: {ex}")
                pause()
            return "open"

        if act.startswith("Выбрать"):
            return "choose"

        return "back"

    def _work_with_chosen_image(self, img_path: Path) -> str:
        while True:
            print()
            action = inquirer.select(
                message=f"Выбрано: {img_path.name}. Действие:",
                choices=["Внедрить (Embed)", "Извлечь (Extract)", "Назад", "Выход"],
                default="Внедрить (Embed)",
            ).execute()

            if action == "Выход":
                return "exit"
            if action == "Назад":
                return "pick_image"

            if action.startswith("Внедрить"):
                nxt = self._embed(img_path)
                if nxt == "pick_image":
                    return "pick_image"
                continue

            if action.startswith("Извлечь"):
                nxt = self._extract(img_path)
                if nxt == "pick_image":
                    return "pick_image"
                continue

    def _choose_algorithm(self) -> AlgorithmChoice | None:
        sections: dict[str, tuple[Literal["text", "watermark"], list[SteganoBase]]] = {}
        if self.text_algorithms:
            sections["Text steganography (скрытие текста)"] = (
                "text",
                list(self.text_algorithms),
            )
        if self.watermark_algorithms:
            sections["Digital watermarking (ЦВЗ / watermark)"] = (
                "watermark",
                list(self.watermark_algorithms),
            )

        section = inquirer.select(
            message="Раздел методов:",
            choices=list(sections.keys()) + ["Назад"],
        ).execute()

        if section == "Назад":
            return None

        kind, algorithms = sections[section]
        amap = {a.get_name(): a for a in algorithms}
        choice = inquirer.select(
            message="Выбор метода:",
            choices=list(amap.keys()) + ["Назад"],
        ).execute()

        if choice == "Назад":
            return None

        return AlgorithmChoice(kind=kind, algorithm=amap[choice])

    def _ask_algo_kwargs(self, algo: SteganoBase) -> dict[str, Any]:
        kwargs: dict[str, Any] = {}
        for name, parser in (algo.get_arguments_to_setup() or {}).items():
            kwargs[name] = ask_parsed(name, parser)
        return kwargs

    @staticmethod
    def _ask_existing_file(prompt: str) -> Path:
        def parse_file_path(text: str) -> Path:
            cleaned_text = text.strip().strip("'\"")
            path = Path(cleaned_text).expanduser()
            
            if not path.exists():
                raise ValueError(f"Путь не существует: {path}")
            if not path.is_file():
                raise ValueError(f"Путь ведет не на файл: {path}")
                
            return path

        # Передаем функцию валидации в ask_parsed
        return ask_parsed(prompt, parse_file_path)

    @staticmethod
    def _load_image(path: Path) -> Image.Image:
        img = Image.open(path)
        img.load()
        return img

    def _embed(self, img_path: Path) -> str:
        chosen = self._choose_algorithm()
        if chosen is None:
            return "image_actions"

        try:
            kwargs = self._ask_algo_kwargs(chosen.algorithm)
            cover = self._load_image(img_path)

            original_message: str | None = None
            original_watermark: Image.Image | None = None
            bits_embedded = None

            if chosen.kind == "text":
                algo = chosen.algorithm
                if not isinstance(algo, SteganoTextBase):
                    raise TypeError("Выбранный метод не является SteganoTextBase")

                message = ask_parsed("message (сообщение)", str)
                stego_img, extra = algo.embed(cover, message, **kwargs)
                original_message = message

            else:
                algo = chosen.algorithm
                if not isinstance(algo, SteganoWatermarkBase):
                    raise TypeError("Выбранный метод не является SteganoWatermarkBase")

                watermark_path = self._ask_existing_file(
                    "watermark_path (путь к изображению-ЦВЗ)"
                )
                watermark = self._load_image(watermark_path)
                stego_img, extra = algo.embed(cover, watermark, **kwargs)
                if isinstance(extra, dict) and isinstance(extra.get("prepared_watermark"), Image.Image):
                    original_watermark = extra["prepared_watermark"]
                else:
                    original_watermark = watermark

            out_path = unique_out_path(
                self.config.images_dir,
                prefix="stego",
                base_stem=img_path.stem,
                ext=".png",
            )
            stego_img.save(out_path)

            if isinstance(extra, dict):
                if "bits_embedded" in extra:
                    bits_embedded = int(extra["bits_embedded"])
                elif "bytes_embedded" in extra:
                    bits_embedded = int(extra["bytes_embedded"]) * 8

            print(f"\nИзображение сохранено: {out_path}\n")

            while True:
                post = inquirer.select(
                    message="Дальше:",
                    choices=[
                        "Открыть (open)",
                        "Показать метрики (metrics)",
                        "Показать гистограммы (hist)",
                        "Продолжить",
                        "Вернуться",
                    ],
                    default="Продолжить",
                ).execute()

                if post.startswith("Открыть"):
                    try:
                        open_in_viewer(out_path)
                    except Exception as ex:
                        print_error(f"Не удалось открыть: {ex}")
                        pause()
                    continue

                if post.startswith("Показать метрики"):
                    extracted = None
                    try:
                        extracted = chosen.algorithm.extract(stego_img, **kwargs)
                    except Exception:
                        extracted = None

                    try:
                        pack = compute_metrics_pack(
                            cover,
                            stego_img,
                            bits_embedded=bits_embedded,
                            original_message=original_message,
                            extracted_message=(
                                extracted if isinstance(extracted, str) else None
                            ),
                            original_watermark=original_watermark,
                            extracted_watermark=(
                                extracted if isinstance(extracted, Image.Image) else None
                            ),
                        )
                        print()
                        print_success(
                            "METRICS:\n"
                            f"MSE:  {pack.mse}\n"
                            f"PSNR: {pack.psnr}\n"
                            f"RMSE: {pack.rmse}\n"
                            f"SSIM: {pack.ssim}\n"
                            f"EC(bpp): {pack.ec_bpp}\n"
                            f"BER: {pack.ber}\n"
                            f"NCC: {pack.ncc}\n"
                        )
                        pause()
                    except NotImplementedError as ex:
                        print_error(str(ex))
                        pause()
                    except Exception as ex:
                        print_error(f"Ошибка метрик: {ex}")
                        pause()
                    continue

                if post.startswith("Показать гистограммы"):
                    choice = inquirer.select(
                        message="Графики:",
                        choices=[
                            "Brightness histogram (L)",
                            "Blue histogram (B)",
                            "RGB histograms (R/G/B)",
                            "Abs diff histogram (L)",
                            "Abs diff histogram (B)",
                            "Назад",
                        ],
                        default="Brightness histogram (L)",
                    ).execute()

                    if choice == "Назад":
                        continue

                    try:
                        if choice == "Brightness histogram (L)":
                            show_hist_brightness(cover, stego_img)
                        elif choice == "Blue histogram (B)":
                            show_hist_channel(cover, stego_img, channel=2, name="Blue")
                        elif choice == "RGB histograms (R/G/B)":
                            show_hist_rgb(cover, stego_img)
                        elif choice == "Abs diff histogram (L)":
                            show_abs_diff_hist_brightness(cover, stego_img)
                        elif choice == "Abs diff histogram (B)":
                            show_abs_diff_hist_channel(
                                cover, stego_img, channel=2, name="Blue"
                            )
                    except Exception as ex:
                        print_error(f"Ошибка построения графика: {ex}")
                        pause()

                    continue

                if post == "Вернуться":
                    return "image_actions"

                return "pick_image"

        except Exception as ex:
            print()
            print_error(f"Ошибка embed: {ex}")
            pause()
            return "image_actions"

    def _show_or_save_extracted_result(self, extracted: Any, img_path: Path) -> None:
        if isinstance(extracted, Image.Image):
            out_img = unique_out_path(
                self.config.results_dir,
                prefix="extract",
                base_stem=img_path.stem,
                ext=".png",
            )
            extracted.save(out_img)
            print(f"\nИзвлечённое изображение сохранено: {out_img}\n")
            return

        if isinstance(extracted, np.ndarray):
            arr = np.clip(extracted, 0, 255).astype(np.uint8)
            out_img = unique_out_path(
                self.config.results_dir,
                prefix="extract",
                base_stem=img_path.stem,
                ext=".png",
            )
            Image.fromarray(arr).save(out_img)
            print(f"\nИзвлечённая матрица сохранена как изображение: {out_img}\n")
            return

        msg = str(extracted)

        print("\nИзвлечённое сообщение:\n")

        if len(msg) <= 100:
            print_success(msg)
            print()
            return

        preview = msg[:100]
        print_success(preview + "...")
        out_txt = unique_out_path(
            self.config.results_dir,
            prefix="extract",
            base_stem=img_path.stem,
            ext=".txt",
        )
        out_txt.write_text(msg, encoding="utf-8")
        print(f"\nПолный текст сохранён в: {out_txt}\n")

    def _extract(self, img_path: Path) -> str:
        chosen = self._choose_algorithm()
        if chosen is None:
            return "image_actions"

        try:
            kwargs = self._ask_algo_kwargs(chosen.algorithm)
            stego = self._load_image(img_path)
            extracted = chosen.algorithm.extract(stego, **kwargs)

            self._show_or_save_extracted_result(extracted, img_path)

            post = inquirer.select(
                message="Дальше:",
                choices=["Продолжить", "Вернуться"],
                default="Продолжить",
            ).execute()

            if post == "Вернуться":
                return "image_actions"
            return "pick_image"

        except Exception as ex:
            print()
            print_error(f"Ошибка extract: {ex}")
            pause()
            return "image_actions"
