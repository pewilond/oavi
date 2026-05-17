from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


VARIANT = 17
METHOD = "LBP"
FEATURE_NAME = "H(LBP)"
BRIGHTNESS_METHOD = "Степенное преобразование"
GAMMA = 0.65
MAX_SIZE = 640

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent

INPUT_IMAGES = [
    ("city", ROOT_DIR / "lab1" / "beautiful-tranquil-scene-the-city-of-amsterdam-free-photo.png"),
    ("page_01", ROOT_DIR / "lab4" / "source_png" / "01.png"),
    ("page_02", ROOT_DIR / "lab4" / "source_png" / "02.png"),
]

SOURCE_DIR = BASE_DIR / "source_png"
GRAY_DIR = BASE_DIR / "gray_png"
CONTRAST_DIR = BASE_DIR / "contrast_png"
LBP_DIR = BASE_DIR / "lbp_png"
HIST_DIR = BASE_DIR / "histograms_png"
PREVIEW_DIR = BASE_DIR / "preview_png"
FEATURES_PATH = BASE_DIR / "lbp_features_variant17.csv"
COMPARE_PATH = BASE_DIR / "texture_compare_variant17.csv"


@dataclass
class ImageResult:
    name: str
    source_path: Path
    gray_path: Path
    contrast_color_path: Path
    contrast_gray_path: Path
    lbp_original_path: Path
    lbp_contrast_path: Path
    brightness_hist_path: Path
    lbp_hist_path: Path
    entropy_original: float
    entropy_contrast: float
    uniformity_original: float
    uniformity_contrast: float
    histogram_distance: float


def ensure_dirs(paths: Iterable[Path]) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def clear_files(path: Path, pattern: str) -> None:
    for file in path.glob(pattern):
        file.unlink()


def load_rgb(path: Path) -> Image.Image:
    image = Image.open(path).convert("RGB")
    image.thumbnail((MAX_SIZE, MAX_SIZE), Image.Resampling.LANCZOS)
    return image


def rgb_to_hsl(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r = rgb[:, :, 0]
    g = rgb[:, :, 1]
    b = rgb[:, :, 2]

    maxc = rgb.max(axis=2)
    minc = rgb.min(axis=2)
    l = (maxc + minc) / 2.0

    s = np.zeros_like(l)
    h = np.zeros_like(l)
    diff = maxc - minc
    active = diff > 1e-8

    s[active] = np.where(
        l[active] < 0.5,
        diff[active] / (maxc[active] + minc[active]),
        diff[active] / (2.0 - maxc[active] - minc[active]),
    )

    red = active & (maxc == r)
    green = active & (maxc == g)
    blue = active & (maxc == b)
    h[red] = ((g[red] - b[red]) / diff[red]) % 6.0
    h[green] = (b[green] - r[green]) / diff[green] + 2.0
    h[blue] = (r[blue] - g[blue]) / diff[blue] + 4.0
    h /= 6.0
    return h, s, l


def hsl_to_rgb(h: np.ndarray, s: np.ndarray, l: np.ndarray) -> np.ndarray:
    c = (1.0 - np.abs(2.0 * l - 1.0)) * s
    hp = h * 6.0
    x = c * (1.0 - np.abs(hp % 2.0 - 1.0))

    zeros = np.zeros_like(h)
    rp = zeros.copy()
    gp = zeros.copy()
    bp = zeros.copy()

    masks = [
        (0 <= hp) & (hp < 1),
        (1 <= hp) & (hp < 2),
        (2 <= hp) & (hp < 3),
        (3 <= hp) & (hp < 4),
        (4 <= hp) & (hp < 5),
        (5 <= hp) & (hp <= 6),
    ]
    values = [(c, x, zeros), (x, c, zeros), (zeros, c, x), (zeros, x, c), (x, zeros, c), (c, zeros, x)]

    for mask, (rv, gv, bv) in zip(masks, values, strict=True):
        rp[mask] = rv[mask]
        gp[mask] = gv[mask]
        bp[mask] = bv[mask]

    m = l - c / 2.0
    return np.dstack([rp + m, gp + m, bp + m])


def power_contrast(rgb_u8: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rgb = rgb_u8.astype(float) / 255.0
    h, s, l = rgb_to_hsl(rgb)
    new_l = np.clip(l ** GAMMA, 0.0, 1.0)
    new_rgb = hsl_to_rgb(h, s, new_l)
    return (np.clip(new_rgb * 255.0, 0, 255).astype(np.uint8), l, new_l)


def to_gray_from_l(l: np.ndarray) -> np.ndarray:
    return np.clip(l * 255.0, 0, 255).astype(np.uint8)


def lbp(gray: np.ndarray) -> np.ndarray:
    center = gray[1:-1, 1:-1]
    result = np.zeros(center.shape, dtype=np.uint8)
    neighbors = [
        gray[:-2, :-2],
        gray[:-2, 1:-1],
        gray[:-2, 2:],
        gray[1:-1, 2:],
        gray[2:, 2:],
        gray[2:, 1:-1],
        gray[2:, :-2],
        gray[1:-1, :-2],
    ]

    for bit, neighbor in enumerate(neighbors):
        result |= ((neighbor >= center).astype(np.uint8) << bit)
    return result


def normalized_hist(values: np.ndarray, bins: int = 256) -> np.ndarray:
    hist = np.bincount(values.ravel(), minlength=bins).astype(float)
    total = hist.sum()
    return hist / total if total else hist


def entropy(hist: np.ndarray) -> float:
    active = hist[hist > 0]
    return float(-(active * np.log2(active)).sum())


def uniformity(hist: np.ndarray) -> float:
    return float((hist ** 2).sum())


def chi_square_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(0.5 * np.sum(((a - b) ** 2) / (a + b + 1e-12)))


def save_histograms(name: str, gray: np.ndarray, gray_contrast: np.ndarray, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    axes[0].hist(gray.ravel(), bins=256, range=(0, 255), color="black")
    axes[0].set_title("До")
    axes[1].hist(gray_contrast.ravel(), bins=256, range=(0, 255), color="black")
    axes[1].set_title("После")
    for ax in axes:
        ax.set_xlabel("Яркость")
        ax.set_ylabel("Количество")
        ax.grid(axis="y", linestyle=":", alpha=0.35)
    fig.suptitle(f"Гистограмма яркости: {name}")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_lbp_histograms(name: str, hist_a: np.ndarray, hist_b: np.ndarray, path: Path) -> None:
    x = np.arange(256)
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    axes[0].bar(x, hist_a, color="black", width=1.0)
    axes[0].set_title("LBP до")
    axes[1].bar(x, hist_b, color="black", width=1.0)
    axes[1].set_title("LBP после")
    for ax in axes:
        ax.set_ylabel("H(LBP)")
        ax.grid(axis="y", linestyle=":", alpha=0.35)
    axes[1].set_xlabel("Код LBP")
    fig.suptitle(f"Гистограмма LBP: {name}")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_lbp_image(codes: np.ndarray, path: Path) -> None:
    Image.fromarray(codes, mode="L").save(path)


def save_overview(results: list[ImageResult]) -> None:
    for result in results:
        files = [
            result.source_path,
            result.gray_path,
            result.contrast_color_path,
            result.contrast_gray_path,
            result.lbp_original_path,
            result.lbp_contrast_path,
        ]
        titles = ["Исходное", "Полутоновое", "Контраст", "Контраст L", "LBP до", "LBP после"]
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        axes = axes.ravel()
        for ax, file, title in zip(axes, files, titles, strict=True):
            image = Image.open(file)
            cmap = "gray" if image.mode == "L" else None
            ax.imshow(image, cmap=cmap, vmin=0, vmax=255)
            ax.set_title(title)
            ax.set_xticks([])
            ax.set_yticks([])
        fig.suptitle(result.name)
        fig.tight_layout()
        fig.savefig(PREVIEW_DIR / f"{result.name}_overview.png", dpi=150)
        plt.close(fig)


def save_features_csv(rows: list[dict[str, str | int | float]]) -> None:
    fieldnames = ["image", "stage", "entropy", "uniformity"] + [f"lbp_{i}" for i in range(256)]
    with FEATURES_PATH.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=";")
        writer.writeheader()
        writer.writerows(rows)


def save_compare_csv(results: list[ImageResult]) -> None:
    with COMPARE_PATH.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image",
                "entropy_original",
                "entropy_contrast",
                "uniformity_original",
                "uniformity_contrast",
                "lbp_chi_square_distance",
            ],
            delimiter=";",
        )
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "image": result.name,
                    "entropy_original": round(result.entropy_original, 6),
                    "entropy_contrast": round(result.entropy_contrast, 6),
                    "uniformity_original": round(result.uniformity_original, 6),
                    "uniformity_contrast": round(result.uniformity_contrast, 6),
                    "lbp_chi_square_distance": round(result.histogram_distance, 6),
                }
            )


def process_image(name: str, input_path: Path, feature_rows: list[dict[str, str | int | float]]) -> ImageResult:
    source = load_rgb(input_path)
    source_path = SOURCE_DIR / f"{name}.png"
    source.save(source_path)

    rgb = np.array(source, dtype=np.uint8)
    contrast_rgb, l, new_l = power_contrast(rgb)
    gray = to_gray_from_l(l)
    gray_contrast = to_gray_from_l(new_l)

    gray_path = GRAY_DIR / f"{name}_gray.png"
    contrast_color_path = CONTRAST_DIR / f"{name}_contrast.png"
    contrast_gray_path = CONTRAST_DIR / f"{name}_contrast_gray.png"
    Image.fromarray(gray, mode="L").save(gray_path)
    Image.fromarray(contrast_rgb, mode="RGB").save(contrast_color_path)
    Image.fromarray(gray_contrast, mode="L").save(contrast_gray_path)

    lbp_original = lbp(gray)
    lbp_contrast = lbp(gray_contrast)
    lbp_original_path = LBP_DIR / f"{name}_lbp_original.png"
    lbp_contrast_path = LBP_DIR / f"{name}_lbp_contrast.png"
    save_lbp_image(lbp_original, lbp_original_path)
    save_lbp_image(lbp_contrast, lbp_contrast_path)

    hist_original = normalized_hist(lbp_original)
    hist_contrast = normalized_hist(lbp_contrast)
    entropy_original = entropy(hist_original)
    entropy_contrast = entropy(hist_contrast)
    uniformity_original = uniformity(hist_original)
    uniformity_contrast = uniformity(hist_contrast)
    distance = chi_square_distance(hist_original, hist_contrast)

    for stage, hist, ent, uni in [
        ("original", hist_original, entropy_original, uniformity_original),
        ("contrast", hist_contrast, entropy_contrast, uniformity_contrast),
    ]:
        row: dict[str, str | int | float] = {
            "image": name,
            "stage": stage,
            "entropy": round(ent, 6),
            "uniformity": round(uni, 6),
        }
        for i, value in enumerate(hist):
            row[f"lbp_{i}"] = round(float(value), 8)
        feature_rows.append(row)

    brightness_hist_path = HIST_DIR / f"{name}_brightness_hist.png"
    lbp_hist_path = HIST_DIR / f"{name}_lbp_hist.png"
    save_histograms(name, gray, gray_contrast, brightness_hist_path)
    save_lbp_histograms(name, hist_original, hist_contrast, lbp_hist_path)

    return ImageResult(
        name=name,
        source_path=source_path,
        gray_path=gray_path,
        contrast_color_path=contrast_color_path,
        contrast_gray_path=contrast_gray_path,
        lbp_original_path=lbp_original_path,
        lbp_contrast_path=lbp_contrast_path,
        brightness_hist_path=brightness_hist_path,
        lbp_hist_path=lbp_hist_path,
        entropy_original=entropy_original,
        entropy_contrast=entropy_contrast,
        uniformity_original=uniformity_original,
        uniformity_contrast=uniformity_contrast,
        histogram_distance=distance,
    )


def run_pipeline() -> dict[str, str | int | float]:
    ensure_dirs([SOURCE_DIR, GRAY_DIR, CONTRAST_DIR, LBP_DIR, HIST_DIR, PREVIEW_DIR])
    for path in [SOURCE_DIR, GRAY_DIR, CONTRAST_DIR, LBP_DIR, HIST_DIR, PREVIEW_DIR]:
        clear_files(path, "*.png")

    feature_rows: list[dict[str, str | int | float]] = []
    results = [process_image(name, path, feature_rows) for name, path in INPUT_IMAGES]
    save_features_csv(feature_rows)
    save_compare_csv(results)
    save_overview(results)

    return {
        "variant": VARIANT,
        "method": METHOD,
        "feature": FEATURE_NAME,
        "brightness_method": BRIGHTNESS_METHOD,
        "gamma": GAMMA,
        "image_count": len(results),
        "features_csv": str(FEATURES_PATH),
        "compare_csv": str(COMPARE_PATH),
        "preview_dir": str(PREVIEW_DIR),
    }


if __name__ == "__main__":
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    summary = run_pipeline()
    for key, value in summary.items():
        print(f"{key}: {value}")
