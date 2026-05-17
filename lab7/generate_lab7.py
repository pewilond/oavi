from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont


VARIANT = 17
ALPHABET_NAME = "Казахские строчные буквы"
ALPHABET = list("аәбвгғдеёжзийкқлмнңоөпрстуұүфхһцчшщъыіьэюя")
PHRASE = "сәулем мен сені сүйемін"

FONT_NAME = "Arial"
FONT_PATH = Path(r"C:\Windows\Fonts\arial.ttf")
FONT_SIZE = 52
EXPERIMENT_FONT_SIZE = 58
THRESHOLD = 200
PROFILE_THRESHOLD = 2
CANVAS_MARGIN = 30
FEATURE_WEIGHTS = np.array([0.1, 1.0, 1.0, 2.0, 2.0], dtype=float)

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
LAB5_DIR = ROOT_DIR / "lab5"
LAB6_DIR = ROOT_DIR / "lab6"

PREVIEW_DIR = BASE_DIR / "preview_png"
EXPERIMENT_DIR = BASE_DIR / "experiment_bmp"
MAIN_HYPOTHESES_PATH = BASE_DIR / "hypotheses_variant17.txt"
MAIN_CSV_PATH = BASE_DIR / "classification_main.csv"
EXPERIMENT_HYPOTHESES_PATH = BASE_DIR / "hypotheses_variant17_font58.txt"
EXPERIMENT_CSV_PATH = BASE_DIR / "classification_experiment_font58.csv"
SUMMARY_PATH = BASE_DIR / "summary_variant17.csv"


@dataclass
class SymbolImage:
    index: int
    symbol: str
    codepoint: str
    image: np.ndarray
    path: Path


@dataclass
class RecognitionResult:
    mode: str
    expected: str
    recognized: str
    errors: int
    accuracy: float
    hypotheses_path: Path
    csv_path: Path
    preview_path: Path


def ensure_dirs(paths: Iterable[Path]) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def clear_files(path: Path, pattern: str) -> None:
    for file in path.glob(pattern):
        file.unlink()


def codepoint_label(symbol: str) -> str:
    return "u" + "_".join(f"{ord(ch):04x}" for ch in symbol)


def load_binary(path: Path) -> np.ndarray:
    arr = np.array(Image.open(path).convert("L"), dtype=np.uint8)
    return np.where(arr < THRESHOLD, 0, 255).astype(np.uint8)


def crop_binary(binary: np.ndarray) -> np.ndarray:
    mask = binary == 0
    ys, xs = np.where(mask)
    if ys.size == 0:
        return binary
    return binary[int(ys.min()): int(ys.max()) + 1, int(xs.min()): int(xs.max()) + 1]


def features(binary: np.ndarray) -> np.ndarray:
    binary = crop_binary(binary)
    mask = binary == 0
    height, width = mask.shape
    mass = int(mask.sum())
    area = height * width

    ys, xs = np.where(mask)
    if mass == 0:
        return np.zeros(5, dtype=float)

    cx = float(xs.mean())
    cy = float(ys.mean())
    inertia_h = float(((ys - cy) ** 2).sum())
    inertia_v = float(((xs - cx) ** 2).sum())

    return np.array(
        [
            mass / area,
            cx / (width - 1) if width > 1 else 0.0,
            cy / (height - 1) if height > 1 else 0.0,
            inertia_h / (mass * (height - 1) ** 2) if height > 1 else 0.0,
            inertia_v / (mass * (width - 1) ** 2) if width > 1 else 0.0,
        ],
        dtype=float,
    )


def similarity(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
    distance = float(np.linalg.norm((vector_a - vector_b) * FEATURE_WEIGHTS))
    return 1.0 / (1.0 + distance)


def load_alphabet() -> list[SymbolImage]:
    records: list[SymbolImage] = []
    for symbol in ALPHABET:
        codepoint = codepoint_label(symbol)
        path = LAB5_DIR / "symbols_png" / f"{codepoint}.png"
        records.append(SymbolImage(0, symbol, codepoint, load_binary(path), path))
    return records


def load_lab6_segments() -> list[SymbolImage]:
    csv_path = LAB6_DIR / "segments_variant17.csv"
    records: list[SymbolImage] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            path = LAB6_DIR / "segmented_symbols_png" / row["image_path"]
            records.append(
                SymbolImage(
                    int(row["index"]),
                    row["symbol"],
                    row["codepoint"],
                    load_binary(path),
                    path,
                )
            )
    return records


def render_phrase(font_size: int) -> np.ndarray:
    font = ImageFont.truetype(str(FONT_PATH), font_size)
    probe = Image.new("L", (10, 10), 255)
    draw = ImageDraw.Draw(probe)
    bbox = draw.textbbox((0, 0), PHRASE, font=font)
    width = bbox[2] - bbox[0] + CANVAS_MARGIN * 2
    height = bbox[3] - bbox[1] + CANVAS_MARGIN * 2

    image = Image.new("L", (width, height), 255)
    draw = ImageDraw.Draw(image)
    draw.text((CANVAS_MARGIN - bbox[0], CANVAS_MARGIN - bbox[1]), PHRASE, font=font, fill=0)
    return crop_binary(np.where(np.array(image) < THRESHOLD, 0, 255).astype(np.uint8))


def find_runs(active: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for i, value in enumerate(active):
        if value and start is None:
            start = i
        elif not value and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(active)))
    return runs


def segment_binary(binary: np.ndarray) -> list[np.ndarray]:
    mask = binary == 0
    segments: list[np.ndarray] = []
    for left, right in find_runs(mask.sum(axis=0) > PROFILE_THRESHOLD):
        part = mask[:, left:right]
        ys, xs = np.where(part)
        if ys.size == 0:
            continue
        top = int(ys.min())
        bottom = int(ys.max()) + 1
        real_left = left + int(xs.min())
        real_right = left + int(xs.max()) + 1
        segments.append(binary[top:bottom, real_left:real_right])
    return segments


def make_experiment_segments() -> list[SymbolImage]:
    binary = render_phrase(EXPERIMENT_FONT_SIZE)
    bmp_path = EXPERIMENT_DIR / "phrase_variant17_font58.bmp"
    Image.fromarray(binary, mode="L").save(bmp_path)

    symbols = [ch for ch in PHRASE if ch != " "]
    parts = segment_binary(binary)
    if len(parts) != len(symbols):
        raise ValueError(f"Получено {len(parts)} сегментов вместо {len(symbols)}")

    records: list[SymbolImage] = []
    for index, (symbol, part) in enumerate(zip(symbols, parts, strict=True), start=1):
        records.append(SymbolImage(index, symbol, codepoint_label(symbol), part, bmp_path))
    return records


def classify_segments(
    mode: str,
    segments: list[SymbolImage],
    alphabet: list[SymbolImage],
    hypotheses_path: Path,
    csv_path: Path,
    preview_path: Path,
) -> RecognitionResult:
    alphabet_vectors = {item.symbol: features(item.image) for item in alphabet}
    expected = "".join(item.symbol for item in segments)
    recognized_chars: list[str] = []
    all_rows: list[dict[str, str | int | float]] = []
    text_lines: list[str] = []
    preview_rows: list[tuple[SymbolImage, str, float]] = []

    for segment in segments:
        vector = features(segment.image)
        hypotheses = sorted(
            ((item.symbol, similarity(vector, alphabet_vectors[item.symbol])) for item in alphabet),
            key=lambda item: item[1],
            reverse=True,
        )
        best_symbol, best_score = hypotheses[0]
        recognized_chars.append(best_symbol)
        preview_rows.append((segment, best_symbol, best_score))

        short = ", ".join(f"('{symbol}', {score:.4f})" for symbol, score in hypotheses)
        text_lines.append(f"{segment.index}: [{short}]")

        row: dict[str, str | int | float] = {
            "index": segment.index,
            "expected": segment.symbol,
            "recognized": best_symbol,
            "best_similarity": round(best_score, 6),
            "is_correct": int(best_symbol == segment.symbol),
        }
        for rank, (symbol, score) in enumerate(hypotheses[:5], start=1):
            row[f"top{rank}_symbol"] = symbol
            row[f"top{rank}_similarity"] = round(score, 6)
        all_rows.append(row)

    hypotheses_path.write_text("\n".join(text_lines), encoding="utf-8")
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()), delimiter=";")
        writer.writeheader()
        writer.writerows(all_rows)

    recognized = "".join(recognized_chars)
    errors = sum(a != b for a, b in zip(expected, recognized, strict=True))
    accuracy = 100.0 * (len(expected) - errors) / len(expected)
    save_preview(preview_rows, preview_path, mode)

    return RecognitionResult(mode, expected, recognized, errors, accuracy, hypotheses_path, csv_path, preview_path)


def save_preview(rows: list[tuple[SymbolImage, str, float]], path: Path, title: str) -> None:
    cols = 5
    rows_count = int(np.ceil(len(rows) / cols))
    fig, axes = plt.subplots(rows_count, cols, figsize=(cols * 2.5, rows_count * 2.4))
    axes = np.atleast_1d(axes).ravel()

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor("#f2f2f2")

    for ax, (segment, predicted, score) in zip(axes, rows):
        ax.imshow(segment.image, cmap="gray", vmin=0, vmax=255)
        color = "green" if predicted == segment.symbol else "crimson"
        ax.set_title(f"{segment.index}: {segment.symbol}->{predicted}\n{score:.3f}", color=color, fontsize=10)

    for ax in axes[len(rows):]:
        ax.axis("off")

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def restore_spaces(compact: str) -> str:
    result: list[str] = []
    pos = 0
    for ch in PHRASE:
        if ch == " ":
            result.append(" ")
        else:
            result.append(compact[pos])
            pos += 1
    return "".join(result)


def write_summary(results: list[RecognitionResult]) -> None:
    with SUMMARY_PATH.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "mode",
                "expected",
                "recognized",
                "recognized_with_spaces",
                "errors",
                "accuracy_percent",
                "hypotheses_path",
                "csv_path",
            ],
            delimiter=";",
        )
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "mode": result.mode,
                    "expected": result.expected,
                    "recognized": result.recognized,
                    "recognized_with_spaces": restore_spaces(result.recognized),
                    "errors": result.errors,
                    "accuracy_percent": round(result.accuracy, 2),
                    "hypotheses_path": result.hypotheses_path.name,
                    "csv_path": result.csv_path.name,
                }
            )


def run_pipeline() -> dict[str, str | int | float]:
    ensure_dirs([PREVIEW_DIR, EXPERIMENT_DIR])
    clear_files(PREVIEW_DIR, "*.png")
    clear_files(EXPERIMENT_DIR, "*.bmp")

    alphabet = load_alphabet()
    main_segments = load_lab6_segments()
    experiment_segments = make_experiment_segments()

    main = classify_segments(
        "Исходная строка",
        main_segments,
        alphabet,
        MAIN_HYPOTHESES_PATH,
        MAIN_CSV_PATH,
        PREVIEW_DIR / "classification_main.png",
    )
    experiment = classify_segments(
        f"Эксперимент, кегль {EXPERIMENT_FONT_SIZE}",
        experiment_segments,
        alphabet,
        EXPERIMENT_HYPOTHESES_PATH,
        EXPERIMENT_CSV_PATH,
        PREVIEW_DIR / "classification_experiment_font58.png",
    )
    write_summary([main, experiment])

    return {
        "variant": VARIANT,
        "alphabet_name": ALPHABET_NAME,
        "phrase": PHRASE,
        "font_name": FONT_NAME,
        "font_size": FONT_SIZE,
        "experiment_font_size": EXPERIMENT_FONT_SIZE,
        "feature_weights": "mass=0.1, centroid_x=1, centroid_y=1, inertia_h=2, inertia_v=2",
        "main_recognized": restore_spaces(main.recognized),
        "main_errors": main.errors,
        "main_accuracy_percent": round(main.accuracy, 2),
        "experiment_recognized": restore_spaces(experiment.recognized),
        "experiment_errors": experiment.errors,
        "experiment_accuracy_percent": round(experiment.accuracy, 2),
        "summary_path": str(SUMMARY_PATH),
        "main_hypotheses_path": str(MAIN_HYPOTHESES_PATH),
        "experiment_hypotheses_path": str(EXPERIMENT_HYPOTHESES_PATH),
    }


if __name__ == "__main__":
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    summary = run_pipeline()
    for key, value in summary.items():
        print(f"{key}: {value}")
