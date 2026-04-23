from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from InquirerPy import inquirer

from algorithms.stegano_base import SteganoBase
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


class SteganoTuiApp:
    def __init__(self, algorithms: list[SteganoBase], config: TuiConfig | None = None):
        self.algorithms = algorithms
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

    def _choose_algorithm(self) -> SteganoBase | None:
        amap = {a.get_name(): a for a in self.algorithms}
        choice = inquirer.select(
            message="Выбор метода:",
            choices=list(amap.keys()) + ["Назад"],
        ).execute()
        if choice == "Назад":
            return None
        return amap[choice]

    def _ask_algo_kwargs(self, algo: SteganoBase) -> dict[str, Any]:
        kwargs: dict[str, Any] = {}
        params = algo.get_arguments_to_setup() or {}
        for name, parser in params.items():
            kwargs[name] = ask_parsed(name, parser)
        return kwargs

    def _embed(self, img_path: Path) -> str:
        algo = self._choose_algorithm()
        if algo is None:
            return "image_actions"

        try:
            kwargs = self._ask_algo_kwargs(algo)
            message = ask_parsed("message (сообщение)", str)

            cover = algo.load_image(img_path)
            stego_img, extra = algo.embed(cover, message, **kwargs)

            out_path = unique_out_path(
                self.config.images_dir,
                prefix="stego",
                base_stem=img_path.stem,
                ext=".png",
            )
            stego_img.save(out_path)

            bits_embedded = None
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
                        extracted = algo.extract(stego_img, **kwargs)
                    except Exception:
                        extracted = None

                    try:
                        pack = compute_metrics_pack(
                            cover,
                            stego_img,
                            bits_embedded=bits_embedded,
                            original_message=message,
                            extracted_message=extracted,
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

    def _extract(self, img_path: Path) -> str:
        MAX_PREVIEW = 100

        algo = self._choose_algorithm()
        if algo is None:
            return "image_actions"

        try:
            kwargs = self._ask_algo_kwargs(algo)
            msg = algo.extract_path(img_path, **kwargs)

            print("\nИзвлечённое сообщение:\n")

            if len(msg) <= MAX_PREVIEW:
                print_success(msg)
                print()
            else:
                preview = msg[:MAX_PREVIEW]
                print_success(preview + "...")
                out_txt = unique_out_path(
                    self.config.results_dir,
                    prefix="extract",
                    base_stem=img_path.stem,
                    ext=".txt",
                )
                out_txt.write_text(msg, encoding="utf-8")
                print(f"\nПолный текст сохранён в: {out_txt}\n")

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
