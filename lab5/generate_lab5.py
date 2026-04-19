from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
from PIL import Image, ImageDraw, ImageFont


VARIANT = 17
ALPHABET_NAME = "Казахские строчные буквы"
ALPHABET = list("аәбвгғдеёжзийкқлмнңоөпрстуұүфхһцчшщъыіьэюя")
FONT_NAME = "Arial"
FONT_PATH = Path(r"C:\Windows\Fonts\arial.ttf")
FONT_SIZE = 52
CANVAS_SIZE = (256, 256)
THRESHOLD = 200
PADDING = 4

BASE_DIR = Path(__file__).resolve().parent
SYMBOL_DIR = BASE_DIR / "symbols_png"
PROFILE_X_DIR = BASE_DIR / "profiles_x_png"
PROFILE_Y_DIR = BASE_DIR / "profiles_y_png"
PREVIEW_DIR = BASE_DIR / "preview_png"
CSV_PATH = BASE_DIR / "features_variant17.csv"


@dataclass
class SymbolRecord:
    symbol: str
    codepoint: str
    width: int
    height: int
    image_path: Path
    profile_x_path: Path
    profile_y_path: Path
    features: dict[str, float | int | str]


def ensure_dirs(paths: Iterable[Path]) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def clear_pngs(path: Path) -> None:
    for file in path.glob("*.png"):
        file.unlink()


def clear_csv(path: Path) -> None:
    if path.exists():
        path.unlink()


def codepoint_label(symbol: str) -> str:
    return "u" + "_".join(f"{ord(ch):04x}" for ch in symbol)


def load_font() -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(FONT_PATH), FONT_SIZE)


def render_symbol(symbol: str, font: ImageFont.FreeTypeFont) -> np.ndarray:
    image = Image.new("L", CANVAS_SIZE, 255)
    draw = ImageDraw.Draw(image)
    bbox = draw.textbbox((0, 0), symbol, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    x = (CANVAS_SIZE[0] - text_width) // 2 - bbox[0]
    y = (CANVAS_SIZE[1] - text_height) // 2 - bbox[1]
    draw.text((x, y), symbol, font=font, fill=0)

    arr = np.array(image, dtype=np.uint8)
    mask = arr < THRESHOLD
    ys, xs = np.where(mask)
    if ys.size == 0 or xs.size == 0:
        raise ValueError(f"Не удалось отрисовать символ {symbol!r}")

    left = max(int(xs.min()) - PADDING, 0)
    top = max(int(ys.min()) - PADDING, 0)
    right = min(int(xs.max()) + PADDING + 1, arr.shape[1])
    bottom = min(int(ys.max()) + PADDING + 1, arr.shape[0])

    cropped = arr[top:bottom, left:right]
    binary = np.where(cropped < THRESHOLD, 0, 255).astype(np.uint8)
    return binary


def quarter_slices(mask: np.ndarray) -> dict[str, np.ndarray]:
    height, width = mask.shape
    mid_y = height // 2
    mid_x = width // 2
    return {
        "q1_top_left": mask[:mid_y, :mid_x],
        "q2_top_right": mask[:mid_y, mid_x:],
        "q3_bottom_left": mask[mid_y:, :mid_x],
        "q4_bottom_right": mask[mid_y:, mid_x:],
    }


def compute_features(symbol: str, binary: np.ndarray) -> tuple[dict[str, float | int | str], np.ndarray, np.ndarray]:
    mask = (binary == 0).astype(np.uint8)
    height, width = mask.shape
    total_mass = int(mask.sum())

    y_idx, x_idx = np.nonzero(mask)
    if total_mass == 0:
        centroid_x = 0.0
        centroid_y = 0.0
        inertia_horizontal = 0.0
        inertia_vertical = 0.0
    else:
        centroid_x = float(x_idx.mean())
        centroid_y = float(y_idx.mean())
        inertia_horizontal = float(((y_idx - centroid_y) ** 2).sum())
        inertia_vertical = float(((x_idx - centroid_x) ** 2).sum())

    centroid_x_norm = centroid_x / (width - 1) if width > 1 else 0.0
    centroid_y_norm = centroid_y / (height - 1) if height > 1 else 0.0
    inertia_horizontal_norm = (
        inertia_horizontal / (total_mass * (height - 1) ** 2)
        if total_mass > 0 and height > 1
        else 0.0
    )
    inertia_vertical_norm = (
        inertia_vertical / (total_mass * (width - 1) ** 2)
        if total_mass > 0 and width > 1
        else 0.0
    )

    features: dict[str, float | int | str] = {
        "symbol": symbol,
        "codepoint": codepoint_label(symbol),
        "width": width,
        "height": height,
        "mass_total": total_mass,
        "centroid_x": round(centroid_x, 6),
        "centroid_y": round(centroid_y, 6),
        "centroid_x_norm": round(centroid_x_norm, 6),
        "centroid_y_norm": round(centroid_y_norm, 6),
        "inertia_horizontal": round(inertia_horizontal, 6),
        "inertia_vertical": round(inertia_vertical, 6),
        "inertia_horizontal_norm": round(inertia_horizontal_norm, 6),
        "inertia_vertical_norm": round(inertia_vertical_norm, 6),
    }

    for name, quarter in quarter_slices(mask).items():
        mass = int(quarter.sum())
        area = int(quarter.size)
        features[f"{name}_mass"] = mass
        features[f"{name}_specific_mass"] = round(mass / area if area else 0.0, 6)

    profile_x = mask.sum(axis=0).astype(int)
    profile_y = mask.sum(axis=1).astype(int)
    return features, profile_x, profile_y


def save_binary_image(binary: np.ndarray, path: Path) -> None:
    Image.fromarray(binary, mode="L").save(path)


def _integer_step(length: int) -> int:
    return max(1, length // 10)


def save_profile_x(symbol: str, codepoint: str, profile_x: np.ndarray, path: Path) -> None:
    fig_width = max(6, profile_x.size / 8)
    fig, ax = plt.subplots(figsize=(fig_width, 4))
    x = np.arange(profile_x.size)
    ax.bar(x, profile_x, color="black", width=0.9)
    ax.set_title(f"Профиль X: {symbol} ({codepoint})")
    ax.set_xlabel("Столбец")
    ax.set_ylabel("Масса черного")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xticks(np.arange(0, profile_x.size, _integer_step(profile_x.size)))
    ax.grid(axis="y", linestyle=":", alpha=0.35)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_profile_y(symbol: str, codepoint: str, profile_y: np.ndarray, path: Path) -> None:
    fig_height = max(6, profile_y.size / 10)
    fig, ax = plt.subplots(figsize=(6, fig_height))
    y = np.arange(profile_y.size)
    ax.barh(y, profile_y, color="black", height=0.9)
    ax.invert_yaxis()
    ax.set_title(f"Профиль Y: {symbol} ({codepoint})")
    ax.set_xlabel("Масса черного")
    ax.set_ylabel("Строка")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_yticks(np.arange(0, profile_y.size, _integer_step(profile_y.size)))
    ax.grid(axis="x", linestyle=":", alpha=0.35)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_symbol_preview(records: list[SymbolRecord], path: Path, limit: int = 12) -> None:
    shown = records[:limit]
    cols = 4
    rows = int(np.ceil(len(shown) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 3.2))
    axes = np.atleast_1d(axes).ravel()

    for ax in axes:
        ax.set_facecolor("#f2f2f2")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("#666666")
            spine.set_linewidth(0.8)

    for ax, record in zip(axes, shown):
        img = Image.open(record.image_path)
        ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"{record.symbol}\n{record.codepoint}", fontsize=10)

    for ax in axes[len(shown):]:
        ax.axis("off")

    fig.suptitle("Примеры эталонных символов", fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_csv(records: list[SymbolRecord], path: Path) -> None:
    fieldnames = list(records[0].features.keys())
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=";")
        writer.writeheader()
        for record in records:
            writer.writerow(record.features)


def run_pipeline() -> dict[str, str | int]:
    ensure_dirs([SYMBOL_DIR, PROFILE_X_DIR, PROFILE_Y_DIR, PREVIEW_DIR])
    clear_pngs(SYMBOL_DIR)
    clear_pngs(PROFILE_X_DIR)
    clear_pngs(PROFILE_Y_DIR)
    clear_pngs(PREVIEW_DIR)
    clear_csv(CSV_PATH)

    font = load_font()
    records: list[SymbolRecord] = []

    for symbol in ALPHABET:
        codepoint = codepoint_label(symbol)
        binary = render_symbol(symbol, font)
        features, profile_x, profile_y = compute_features(symbol, binary)

        image_path = SYMBOL_DIR / f"{codepoint}.png"
        profile_x_path = PROFILE_X_DIR / f"{codepoint}_x.png"
        profile_y_path = PROFILE_Y_DIR / f"{codepoint}_y.png"

        save_binary_image(binary, image_path)
        save_profile_x(symbol, codepoint, profile_x, profile_x_path)
        save_profile_y(symbol, codepoint, profile_y, profile_y_path)

        records.append(
            SymbolRecord(
                symbol=symbol,
                codepoint=codepoint,
                width=int(binary.shape[1]),
                height=int(binary.shape[0]),
                image_path=image_path,
                profile_x_path=profile_x_path,
                profile_y_path=profile_y_path,
                features=features,
            )
        )

    write_csv(records, CSV_PATH)
    save_symbol_preview(records, PREVIEW_DIR / "symbols_overview.png")

    return {
        "variant": VARIANT,
        "alphabet_name": ALPHABET_NAME,
        "font_name": FONT_NAME,
        "font_size": FONT_SIZE,
        "symbol_count": len(records),
        "csv_path": str(CSV_PATH),
        "symbols_dir": str(SYMBOL_DIR),
        "profiles_x_dir": str(PROFILE_X_DIR),
        "profiles_y_dir": str(PROFILE_Y_DIR),
        "preview_path": str(PREVIEW_DIR / "symbols_overview.png"),
    }


if __name__ == "__main__":
    summary = run_pipeline()
    for key, value in summary.items():
        print(f"{key}: {value}")
