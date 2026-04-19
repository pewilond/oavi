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
PHRASE = "сәулем мен сені сүйемін"

FONT_NAME = "Arial"
FONT_PATH = Path(r"C:\Windows\Fonts\arial.ttf")
FONT_SIZE = 52
LINE_CANVAS_SIZE = (1800, 240)
SYMBOL_CANVAS_SIZE = (256, 256)
THRESHOLD = 200
SYMBOL_PADDING = 4
PROFILE_THRESHOLD = 1
PREVIEW_PADDING = 12

BASE_DIR = Path(__file__).resolve().parent
SOURCE_DIR = BASE_DIR / "source_bmp"
PREVIEW_DIR = BASE_DIR / "preview_png"
PROFILES_DIR = BASE_DIR / "profiles_png"
SEGMENTS_DIR = BASE_DIR / "segmented_symbols_png"
ALPHABET_SYMBOL_DIR = BASE_DIR / "alphabet_symbols_png"
ALPHABET_PROFILE_X_DIR = BASE_DIR / "alphabet_profiles_x_png"
ALPHABET_PROFILE_Y_DIR = BASE_DIR / "alphabet_profiles_y_png"
SEGMENTS_CSV_PATH = BASE_DIR / "segments_variant17.csv"


@dataclass
class SegmentRecord:
    index: int
    symbol: str
    codepoint: str
    bbox: tuple[int, int, int, int]
    image_path: Path


@dataclass
class AlphabetRecord:
    symbol: str
    codepoint: str
    image_path: Path
    profile_x_path: Path
    profile_y_path: Path


def ensure_dirs(paths: Iterable[Path]) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def clear_files(path: Path, pattern: str) -> None:
    for file in path.glob(pattern):
        file.unlink()


def clear_csv(path: Path) -> None:
    if path.exists():
        path.unlink()


def codepoint_label(text: str) -> str:
    return "u" + "_".join(f"{ord(ch):04x}" for ch in text)


def load_font() -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(FONT_PATH), FONT_SIZE)


def render_phrase_line(text: str, font: ImageFont.FreeTypeFont) -> np.ndarray:
    image = Image.new("L", LINE_CANVAS_SIZE, 255)
    draw = ImageDraw.Draw(image)
    bbox = draw.textbbox((0, 0), text, font=font)
    x = 20 - bbox[0]
    y = 20 - bbox[1]
    draw.text((x, y), text, font=font, fill=0)

    arr = np.array(image, dtype=np.uint8)
    mask = arr < THRESHOLD
    ys, xs = np.where(mask)
    if ys.size == 0 or xs.size == 0:
        raise ValueError("Не удалось отрисовать строку для сегментации")

    cropped = arr[int(ys.min()): int(ys.max()) + 1, int(xs.min()): int(xs.max()) + 1]
    return np.where(cropped < THRESHOLD, 0, 255).astype(np.uint8)


def render_symbol(symbol: str, font: ImageFont.FreeTypeFont) -> np.ndarray:
    image = Image.new("L", SYMBOL_CANVAS_SIZE, 255)
    draw = ImageDraw.Draw(image)
    bbox = draw.textbbox((0, 0), symbol, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    x = (SYMBOL_CANVAS_SIZE[0] - text_width) // 2 - bbox[0]
    y = (SYMBOL_CANVAS_SIZE[1] - text_height) // 2 - bbox[1]
    draw.text((x, y), symbol, font=font, fill=0)

    arr = np.array(image, dtype=np.uint8)
    mask = arr < THRESHOLD
    ys, xs = np.where(mask)
    if ys.size == 0 or xs.size == 0:
        raise ValueError(f"Не удалось отрисовать символ {symbol!r}")

    left = max(int(xs.min()) - SYMBOL_PADDING, 0)
    top = max(int(ys.min()) - SYMBOL_PADDING, 0)
    right = min(int(xs.max()) + SYMBOL_PADDING + 1, arr.shape[1])
    bottom = min(int(ys.max()) + SYMBOL_PADDING + 1, arr.shape[0])

    cropped = arr[top:bottom, left:right]
    return np.where(cropped < THRESHOLD, 0, 255).astype(np.uint8)


def save_monochrome_bmp(binary: np.ndarray, path: Path) -> None:
    Image.fromarray(binary, mode="L").convert("1").save(path)


def save_binary_png(binary: np.ndarray, path: Path) -> None:
    Image.fromarray(binary, mode="L").save(path)


def add_preview_padding(binary: np.ndarray, pad: int = PREVIEW_PADDING, bg: int = 235) -> np.ndarray:
    preview = np.full((binary.shape[0] + 2 * pad, binary.shape[1] + 2 * pad), bg, dtype=np.uint8)
    preview[pad:pad + binary.shape[0], pad:pad + binary.shape[1]] = binary
    return preview


def compute_profiles(binary: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = (binary == 0).astype(np.uint8)
    horizontal = mask.sum(axis=1).astype(int)
    vertical = mask.sum(axis=0).astype(int)
    return horizontal, vertical


def _integer_step(length: int) -> int:
    return max(1, length // 10)


def save_vertical_profile(title: str, profile: np.ndarray, path: Path) -> None:
    fig_width = max(8, profile.size / 14)
    fig, ax = plt.subplots(figsize=(fig_width, 4))
    x = np.arange(profile.size)
    ax.bar(x, profile, color="black", width=0.9)
    ax.set_title(title)
    ax.set_xlabel("Столбец")
    ax.set_ylabel("Масса черного")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xticks(np.arange(0, profile.size, _integer_step(profile.size)))
    ax.grid(axis="y", linestyle=":", alpha=0.35)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_horizontal_profile(title: str, profile: np.ndarray, path: Path) -> None:
    fig_height = max(6, profile.size / 14)
    fig, ax = plt.subplots(figsize=(6, fig_height))
    y = np.arange(profile.size)
    ax.barh(y, profile, color="black", height=0.9)
    ax.invert_yaxis()
    ax.set_title(title)
    ax.set_xlabel("Масса черного")
    ax.set_ylabel("Строка")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_yticks(np.arange(0, profile.size, _integer_step(profile.size)))
    ax.grid(axis="x", linestyle=":", alpha=0.35)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def find_runs(active: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None

    for idx, value in enumerate(active):
        if value and start is None:
            start = idx
        elif not value and start is not None:
            runs.append((start, idx))
            start = None

    if start is not None:
        runs.append((start, len(active)))

    return runs


def segment_text(binary: np.ndarray, profile_threshold: int = PROFILE_THRESHOLD) -> list[tuple[int, int, int, int]]:
    mask = (binary == 0).astype(np.uint8)
    boxes: list[tuple[int, int, int, int]] = []
    col_runs = find_runs(mask.sum(axis=0) > profile_threshold)

    for col_left, col_right in col_runs:
        fragment = mask[:, col_left:col_right]
        ys, xs = np.where(fragment)
        if ys.size == 0 or xs.size == 0:
            continue

        left = col_left + int(xs.min())
        top = int(ys.min())
        right = col_left + int(xs.max()) + 1
        bottom = int(ys.max()) + 1
        boxes.append((left, top, right, bottom))

    boxes.sort(key=lambda box: (box[0], box[1]))
    return boxes


def save_segmentation_overlay(binary: np.ndarray, boxes: list[tuple[int, int, int, int]], path: Path) -> None:
    preview = add_preview_padding(binary)
    image = Image.fromarray(preview, mode="L").convert("RGB")
    draw = ImageDraw.Draw(image)

    for index, (left, top, right, bottom) in enumerate(boxes, start=1):
        padded_box = (
            left + PREVIEW_PADDING,
            top + PREVIEW_PADDING,
            right + PREVIEW_PADDING,
            bottom + PREVIEW_PADDING,
        )
        draw.rectangle(padded_box, outline=(220, 20, 60), width=2)
        draw.text((padded_box[0], max(0, padded_box[1] - 16)), str(index), fill=(220, 20, 60))

    image.save(path)


def save_segments_preview(records: list[SegmentRecord], path: Path, cols: int = 5) -> None:
    rows = int(np.ceil(len(records) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.0, rows * 3.0))
    axes = np.atleast_1d(axes).ravel()

    for ax in axes:
        ax.set_facecolor("#f0f0f0")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("#666666")
            spine.set_linewidth(0.8)

    for ax, record in zip(axes, records):
        img = np.array(Image.open(record.image_path).convert("L"))
        ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"{record.index}: {record.symbol}", fontsize=10)

    for ax in axes[len(records):]:
        ax.axis("off")

    fig.suptitle("Сегментированные символы", fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_alphabet_preview(records: list[AlphabetRecord], path: Path, limit: int = 12) -> None:
    shown = records[:limit]
    cols = 4
    rows = int(np.ceil(len(shown) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.0, rows * 3.0))
    axes = np.atleast_1d(axes).ravel()

    for ax in axes:
        ax.set_facecolor("#f0f0f0")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("#666666")
            spine.set_linewidth(0.8)

    for ax, record in zip(axes, shown):
        img = np.array(Image.open(record.image_path).convert("L"))
        ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"{record.symbol}\n{record.codepoint}", fontsize=10)

    for ax in axes[len(shown):]:
        ax.axis("off")

    fig.suptitle("Примеры символов выбранного алфавита", fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_segments_csv(records: list[SegmentRecord], path: Path) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["index", "symbol", "codepoint", "left", "top", "right", "bottom", "width", "height", "image_path"],
            delimiter=";",
        )
        writer.writeheader()
        for record in records:
            left, top, right, bottom = record.bbox
            writer.writerow(
                {
                    "index": record.index,
                    "symbol": record.symbol,
                    "codepoint": record.codepoint,
                    "left": left,
                    "top": top,
                    "right": right,
                    "bottom": bottom,
                    "width": right - left,
                    "height": bottom - top,
                    "image_path": record.image_path.name,
                }
            )


def generate_alphabet_profiles(font: ImageFont.FreeTypeFont) -> list[AlphabetRecord]:
    records: list[AlphabetRecord] = []

    for symbol in ALPHABET:
        codepoint = codepoint_label(symbol)
        binary = render_symbol(symbol, font)
        profile_y, profile_x = compute_profiles(binary)

        image_path = ALPHABET_SYMBOL_DIR / f"{codepoint}.png"
        profile_x_path = ALPHABET_PROFILE_X_DIR / f"{codepoint}_x.png"
        profile_y_path = ALPHABET_PROFILE_Y_DIR / f"{codepoint}_y.png"

        save_binary_png(binary, image_path)
        save_vertical_profile(f"Профиль X: {symbol} ({codepoint})", profile_x, profile_x_path)
        save_horizontal_profile(f"Профиль Y: {symbol} ({codepoint})", profile_y, profile_y_path)

        records.append(
            AlphabetRecord(
                symbol=symbol,
                codepoint=codepoint,
                image_path=image_path,
                profile_x_path=profile_x_path,
                profile_y_path=profile_y_path,
            )
        )

    return records


def run_pipeline() -> dict[str, str | int]:
    ensure_dirs(
        [
            SOURCE_DIR,
            PREVIEW_DIR,
            PROFILES_DIR,
            SEGMENTS_DIR,
            ALPHABET_SYMBOL_DIR,
            ALPHABET_PROFILE_X_DIR,
            ALPHABET_PROFILE_Y_DIR,
        ]
    )

    clear_files(SOURCE_DIR, "*.bmp")
    clear_files(PREVIEW_DIR, "*.png")
    clear_files(PROFILES_DIR, "*.png")
    clear_files(SEGMENTS_DIR, "*.png")
    clear_files(ALPHABET_SYMBOL_DIR, "*.png")
    clear_files(ALPHABET_PROFILE_X_DIR, "*.png")
    clear_files(ALPHABET_PROFILE_Y_DIR, "*.png")
    clear_csv(SEGMENTS_CSV_PATH)

    font = load_font()

    line_binary = render_phrase_line(PHRASE, font)
    source_bmp_path = SOURCE_DIR / "phrase_variant17.bmp"
    save_monochrome_bmp(line_binary, source_bmp_path)
    save_binary_png(add_preview_padding(line_binary), PREVIEW_DIR / "phrase_preview.png")

    horizontal_profile, vertical_profile = compute_profiles(line_binary)
    save_horizontal_profile("Горизонтальный профиль строки", horizontal_profile, PROFILES_DIR / "horizontal_profile.png")
    save_vertical_profile("Вертикальный профиль строки", vertical_profile, PROFILES_DIR / "vertical_profile.png")

    boxes = segment_text(line_binary, profile_threshold=PROFILE_THRESHOLD)
    symbols = [ch for ch in PHRASE if ch != " "]
    if len(boxes) != len(symbols):
        raise ValueError(
            f"Количество сегментов ({len(boxes)}) не совпадает с числом символов без пробелов ({len(symbols)})"
        )

    save_segmentation_overlay(line_binary, boxes, PREVIEW_DIR / "segmentation_boxes.png")

    segment_records: list[SegmentRecord] = []
    for index, (symbol, bbox) in enumerate(zip(symbols, boxes, strict=True), start=1):
        left, top, right, bottom = bbox
        binary_symbol = line_binary[top:bottom, left:right]
        codepoint = codepoint_label(symbol)
        image_path = SEGMENTS_DIR / f"{index:02d}_{codepoint}.png"
        save_binary_png(binary_symbol, image_path)
        segment_records.append(
            SegmentRecord(
                index=index,
                symbol=symbol,
                codepoint=codepoint,
                bbox=bbox,
                image_path=image_path,
            )
        )

    save_segments_csv(segment_records, SEGMENTS_CSV_PATH)
    save_segments_preview(segment_records, PREVIEW_DIR / "segments_overview.png")

    alphabet_records = generate_alphabet_profiles(font)
    save_alphabet_preview(alphabet_records, PREVIEW_DIR / "alphabet_overview.png")

    return {
        "variant": VARIANT,
        "alphabet_name": ALPHABET_NAME,
        "phrase": PHRASE,
        "font_name": FONT_NAME,
        "font_size": FONT_SIZE,
        "profile_threshold": PROFILE_THRESHOLD,
        "segment_count": len(segment_records),
        "source_bmp": str(source_bmp_path),
        "segments_csv": str(SEGMENTS_CSV_PATH),
        "segments_dir": str(SEGMENTS_DIR),
        "phrase_preview": str(PREVIEW_DIR / "phrase_preview.png"),
        "segmentation_preview": str(PREVIEW_DIR / "segmentation_boxes.png"),
        "segments_overview": str(PREVIEW_DIR / "segments_overview.png"),
        "alphabet_overview": str(PREVIEW_DIR / "alphabet_overview.png"),
        "horizontal_profile": str(PROFILES_DIR / "horizontal_profile.png"),
        "vertical_profile": str(PROFILES_DIR / "vertical_profile.png"),
    }


if __name__ == "__main__":
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    summary = run_pipeline()
    for key, value in summary.items():
        print(f"{key}: {value}")
