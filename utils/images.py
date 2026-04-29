from __future__ import annotations

import os
import sys
import subprocess
from pathlib import Path


DEFAULT_ALLOWED_EXT = {".png", ".jpg", ".jpeg", ".webp"}


def ensure_dir(dir_path: Path) -> None:
    dir_path.mkdir(parents=True, exist_ok=True)


def list_images(dir_path: Path, allowed_ext=DEFAULT_ALLOWED_EXT) -> list[Path]:
    ensure_dir(dir_path)
    return sorted(
        [p for p in dir_path.iterdir() if p.is_file() and p.suffix.lower() in allowed_ext],
        key=lambda p: p.name.lower(),
    )


def open_in_viewer(path: Path) -> None:
    path = path.resolve()
    if os.name == "nt":
        os.startfile(str(path))  # type: ignore[attr-defined]
        return
    if sys.platform == "darwin":
        subprocess.run(["open", str(path)], check=False)
        return
    subprocess.run(["xdg-open", str(path)], check=False)


def unique_out_path(dir_path: Path, *, prefix: str, base_stem: str, ext: str = ".png") -> Path:
    ensure_dir(dir_path)
    p = dir_path / f"{prefix}_{base_stem}{ext}"
    if not p.exists():
        return p
    i = 2
    while True:
        candidate = dir_path / f"{prefix}_{base_stem}_{i}{ext}"
        if not candidate.exists():
            return candidate
        i += 1