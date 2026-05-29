from __future__ import annotations

import csv
import math
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


VARIANT = 17
LAB_VARIANT = "Вариант 3. Анализатор речи"
TOKENS = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "plus"]

FRAME_SECONDS = 0.025
HOP_SECONDS = 0.010
SEGMENT_PAD_SECONDS = 0.040
MAX_SILENCE_GAP_SECONDS = 0.180
MIN_SEGMENT_SECONDS = 0.120

FEATURE_FRAME_SECONDS = 0.025
FEATURE_HOP_SECONDS = 0.010
N_FFT = 2048
MEL_BANDS = 32
MFCC_COUNT = 14
MEL_MIN_HZ = 80.0
MEL_MAX_HZ = 8000.0
PREEMPHASIS = 0.97

SPECTROGRAM_WINDOW_SIZE = 2048
SPECTROGRAM_HOP_SIZE = 512
SPECTROGRAM_N_FFT = 4096

BASE_DIR = Path(__file__).resolve().parent
ALPHABET_DIR = BASE_DIR / "alphabet"
PHONE_PATH = ALPHABET_DIR / "phone.wav"
TRUTH_PATHS = [BASE_DIR / "phone_truth.txt", ALPHABET_DIR / "phone_truth.txt"]

SEGMENT_DIR = BASE_DIR / "segments_wav"
SPECTROGRAM_DIR = BASE_DIR / "spectrogram_png"
PREVIEW_DIR = BASE_DIR / "preview_png"

ALPHABET_CSV_PATH = BASE_DIR / "alphabet_metadata_variant17.csv"
SEGMENTS_CSV_PATH = BASE_DIR / "segments_variant17.csv"
RECOGNITION_CSV_PATH = BASE_DIR / "recognition_variant17.csv"
METRICS_CSV_PATH = BASE_DIR / "metrics_variant17.csv"
HYPOTHESES_PATH = BASE_DIR / "hypotheses_variant17.txt"


@dataclass
class WavInfo:
    path: Path
    sample_rate: int
    channels: int
    sample_width: int
    frames: int
    duration: float


@dataclass
class Segment:
    index: int
    start_sample: int
    end_sample: int
    expected: str
    path: Path
    signal: np.ndarray

    @property
    def duration(self) -> float:
        return self.signal.size


@dataclass
class Recognition:
    index: int
    expected: str
    recognized: str
    best_distance: float
    second_distance: float
    confidence: float
    hypotheses: list[tuple[str, float]]


def ensure_dirs(paths: Iterable[Path]) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def clear_files(path: Path, pattern: str) -> None:
    for file in path.glob(pattern):
        file.unlink()


def token_label(token: str) -> str:
    return "+" if token == "plus" else token


def token_filename(token: str) -> str:
    return "plus" if token == "plus" else token


def read_wav_mono(path: Path) -> tuple[np.ndarray, WavInfo]:
    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        sample_rate = wav.getframerate()
        sample_width = wav.getsampwidth()
        frames = wav.getnframes()
        raw = wav.readframes(frames)

    if sample_width != 2:
        raise ValueError(f"{path.name}: ожидается 16-bit PCM, получено {sample_width * 8} bit")

    samples = np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)

    info = WavInfo(
        path=path,
        sample_rate=sample_rate,
        channels=channels,
        sample_width=sample_width,
        frames=frames,
        duration=frames / sample_rate,
    )
    return samples, info


def write_wav_16bit(path: Path, signal: np.ndarray, sample_rate: int) -> None:
    clipped = np.clip(signal, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.tobytes())


def rms(signal: np.ndarray) -> float:
    if signal.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(signal, dtype=np.float64))))


def read_truth() -> list[str]:
    for path in TRUTH_PATHS:
        if path.exists():
            return path.read_text(encoding="utf-8").split()
    raise FileNotFoundError("Не найден phone_truth.txt с ожидаемой последовательностью")


def frame_rms(signal: np.ndarray, sample_rate: int, frame_seconds: float, hop_seconds: float) -> tuple[np.ndarray, np.ndarray]:
    frame_size = max(1, int(frame_seconds * sample_rate))
    hop_size = max(1, int(hop_seconds * sample_rate))
    values: list[float] = []
    starts: list[int] = []

    for start in range(0, max(signal.size - frame_size + 1, 1), hop_size):
        frame = signal[start : start + frame_size]
        if frame.size < frame_size:
            frame = np.pad(frame, (0, frame_size - frame.size))
        values.append(rms(frame))
        starts.append(start)

    return np.array(values, dtype=np.float64), np.array(starts, dtype=int)


def find_runs(active: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, value in enumerate(active):
        if value and start is None:
            start = index
        elif not value and start is not None:
            runs.append((start, index))
            start = None
    if start is not None:
        runs.append((start, len(active)))
    return runs


def merge_runs(runs: list[tuple[int, int]], max_gap_frames: int) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for start, end in runs:
        if merged and start - merged[-1][1] <= max_gap_frames:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    return merged


def segment_phone(signal: np.ndarray, sample_rate: int, expected_tokens: list[str]) -> tuple[list[tuple[int, int]], float]:
    values, starts = frame_rms(signal, sample_rate, FRAME_SECONDS, HOP_SECONDS)
    threshold = max(float(np.percentile(values, 20)) * 2.0, float(np.percentile(values, 95)) * 0.035)
    active = values > threshold
    max_gap_frames = int(MAX_SILENCE_GAP_SECONDS / HOP_SECONDS)
    min_segment_frames = int(MIN_SEGMENT_SECONDS / HOP_SECONDS)
    pad_samples = int(SEGMENT_PAD_SECONDS * sample_rate)
    frame_size = int(FRAME_SECONDS * sample_rate)

    runs = merge_runs(find_runs(active), max_gap_frames=max_gap_frames)
    segments: list[tuple[int, int]] = []
    for run_start, run_end in runs:
        if run_end - run_start < min_segment_frames:
            continue
        start_sample = max(0, int(starts[run_start]) - pad_samples)
        end_sample = min(signal.size, int(starts[run_end - 1]) + frame_size + pad_samples)
        segments.append((start_sample, end_sample))

    if len(segments) != len(expected_tokens):
        print(f"Предупреждение: найдено {len(segments)} сегментов, ожидается {len(expected_tokens)}")

    return segments, threshold


def trim_signal(signal: np.ndarray, sample_rate: int) -> np.ndarray:
    values, starts = frame_rms(signal, sample_rate, 0.020, 0.010)
    threshold = max(float(np.percentile(values, 20)) * 2.5, float(values.max()) * 0.035, 1e-4)
    active = values > threshold
    if not np.any(active):
        return signal

    first = int(np.argmax(active))
    last = int(len(active) - np.argmax(active[::-1]) - 1)
    pad = int(0.030 * sample_rate)
    frame_size = int(0.020 * sample_rate)
    start = max(0, int(starts[first]) - pad)
    end = min(signal.size, int(starts[last]) + frame_size + pad)
    return signal[start:end]


def hz_to_mel(freq: np.ndarray | float) -> np.ndarray | float:
    return 2595.0 * np.log10(1.0 + np.asarray(freq) / 700.0)


def mel_to_hz(mel: np.ndarray | float) -> np.ndarray | float:
    return 700.0 * (10.0 ** (np.asarray(mel) / 2595.0) - 1.0)


def mel_filterbank(sample_rate: int) -> np.ndarray:
    mel_points = np.linspace(hz_to_mel(MEL_MIN_HZ), hz_to_mel(MEL_MAX_HZ), MEL_BANDS + 2)
    hz_points = mel_to_hz(mel_points)
    bins = np.floor((N_FFT + 1) * hz_points / sample_rate).astype(int)
    filters = np.zeros((MEL_BANDS, N_FFT // 2 + 1), dtype=np.float64)

    for index in range(1, MEL_BANDS + 1):
        left, center, right = bins[index - 1], bins[index], bins[index + 1]
        if center <= left:
            center = left + 1
        if right <= center:
            right = center + 1

        for bin_index in range(left, center):
            if 0 <= bin_index < filters.shape[1]:
                filters[index - 1, bin_index] = (bin_index - left) / (center - left)
        for bin_index in range(center, right):
            if 0 <= bin_index < filters.shape[1]:
                filters[index - 1, bin_index] = (right - bin_index) / (right - center)

    return filters


def dct_matrix(input_count: int, output_count: int) -> np.ndarray:
    n = np.arange(input_count)
    k = np.arange(output_count)[:, None]
    matrix = np.cos(np.pi / input_count * (n + 0.5) * k)
    matrix[0] *= np.sqrt(1.0 / input_count)
    matrix[1:] *= np.sqrt(2.0 / input_count)
    return matrix


def extract_features(signal: np.ndarray, sample_rate: int) -> np.ndarray:
    signal = trim_signal(signal, sample_rate)
    current_rms = rms(signal)
    if current_rms > 1e-8:
        signal = signal / current_rms * 0.05

    if signal.size > 1:
        signal = np.append(signal[0], signal[1:] - PREEMPHASIS * signal[:-1])

    frame_size = int(FEATURE_FRAME_SECONDS * sample_rate)
    hop_size = int(FEATURE_HOP_SECONDS * sample_rate)
    if signal.size < frame_size:
        signal = np.pad(signal, (0, frame_size - signal.size))

    window = np.hanning(frame_size)
    filters = mel_filterbank(sample_rate)
    dct = dct_matrix(MEL_BANDS, MFCC_COUNT)
    rows: list[np.ndarray] = []

    for start in range(0, signal.size - frame_size + 1, hop_size):
        frame = signal[start : start + frame_size] * window
        spectrum = np.fft.rfft(frame, n=N_FFT)
        power = (np.abs(spectrum) ** 2) / N_FFT
        mel_energy = np.maximum(power @ filters.T, 1e-10)
        log_mel = np.log(mel_energy)
        mfcc = log_mel @ dct.T
        log_energy = np.log(np.sum(power) + 1e-10)
        rows.append(np.r_[mfcc[1:MFCC_COUNT], log_energy])

    features = np.vstack(rows)
    return (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-6)


def dtw_distance(a: np.ndarray, b: np.ndarray) -> float:
    n, m = len(a), len(b)
    costs = np.full((n + 1, m + 1), np.inf, dtype=np.float64)
    costs[0, 0] = 0.0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            frame_distance = float(np.linalg.norm(a[i - 1] - b[j - 1]))
            costs[i, j] = frame_distance + min(costs[i - 1, j], costs[i, j - 1], costs[i - 1, j - 1])

    return float(costs[n, m] / (n + m))


def load_alphabet() -> tuple[dict[str, np.ndarray], list[dict[str, str | int | float]]]:
    features: dict[str, np.ndarray] = {}
    rows: list[dict[str, str | int | float]] = []
    sample_rate: int | None = None

    for token in TOKENS:
        path = ALPHABET_DIR / f"{token_filename(token)}.wav"
        signal, info = read_wav_mono(path)
        if sample_rate is None:
            sample_rate = info.sample_rate
        elif sample_rate != info.sample_rate:
            raise ValueError("Все образцы алфавита должны иметь одинаковую частоту дискретизации")

        trimmed = trim_signal(signal, info.sample_rate)
        features[token] = extract_features(signal, info.sample_rate)
        rows.append(
            {
                "token": token,
                "label": token_label(token),
                "file": path.name,
                "sample_rate": info.sample_rate,
                "channels": info.channels,
                "duration_s": round(info.duration, 3),
                "trimmed_duration_s": round(trimmed.size / info.sample_rate, 3),
                "rms": round(rms(signal), 8),
            }
        )

    return features, rows


def save_alphabet_csv(rows: list[dict[str, str | int | float]]) -> None:
    with ALPHABET_CSV_PATH.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter=";")
        writer.writeheader()
        writer.writerows(rows)


def save_segments(
    signal: np.ndarray,
    sample_rate: int,
    segment_bounds: list[tuple[int, int]],
    expected_tokens: list[str],
) -> list[Segment]:
    records: list[Segment] = []
    for index, (start, end) in enumerate(segment_bounds, start=1):
        expected = expected_tokens[index - 1] if index - 1 < len(expected_tokens) else ""
        token_part = token_filename(expected) if expected else "unknown"
        segment_signal = signal[start:end]
        path = SEGMENT_DIR / f"{index:02d}_{token_part}.wav"
        write_wav_16bit(path, segment_signal, sample_rate)
        records.append(
            Segment(
                index=index,
                start_sample=start,
                end_sample=end,
                expected=expected,
                path=path,
                signal=segment_signal,
            )
        )
    return records


def recognize_segments(
    segments: list[Segment],
    alphabet_features: dict[str, np.ndarray],
    sample_rate: int,
) -> list[Recognition]:
    recognitions: list[Recognition] = []

    for segment in segments:
        segment_features = extract_features(segment.signal, sample_rate)
        hypotheses = sorted(
            ((token, dtw_distance(segment_features, features)) for token, features in alphabet_features.items()),
            key=lambda item: item[1],
        )
        best_token, best_distance = hypotheses[0]
        second_distance = hypotheses[1][1] if len(hypotheses) > 1 else best_distance
        confidence = max(0.0, 1.0 - best_distance / (second_distance + 1e-12))
        recognitions.append(
            Recognition(
                index=segment.index,
                expected=segment.expected,
                recognized=best_token,
                best_distance=best_distance,
                second_distance=second_distance,
                confidence=confidence,
                hypotheses=hypotheses,
            )
        )

    return recognitions


def save_segments_csv(segments: list[Segment], sample_rate: int) -> None:
    fieldnames = ["index", "expected", "start_s", "end_s", "duration_s", "file"]
    with SEGMENTS_CSV_PATH.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=";")
        writer.writeheader()
        for segment in segments:
            writer.writerow(
                {
                    "index": segment.index,
                    "expected": segment.expected,
                    "start_s": round(segment.start_sample / sample_rate, 3),
                    "end_s": round(segment.end_sample / sample_rate, 3),
                    "duration_s": round((segment.end_sample - segment.start_sample) / sample_rate, 3),
                    "file": segment.path.name,
                }
            )


def save_recognition_csv(recognitions: list[Recognition]) -> None:
    fieldnames = [
        "index",
        "expected",
        "recognized",
        "is_correct",
        "best_distance",
        "second_distance",
        "confidence",
        "top1",
        "top2",
        "top3",
        "top4",
        "top5",
    ]
    with RECOGNITION_CSV_PATH.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=";")
        writer.writeheader()
        for recognition in recognitions:
            row: dict[str, str | int | float] = {
                "index": recognition.index,
                "expected": recognition.expected,
                "recognized": recognition.recognized,
                "is_correct": int(recognition.expected == recognition.recognized),
                "best_distance": round(recognition.best_distance, 6),
                "second_distance": round(recognition.second_distance, 6),
                "confidence": round(recognition.confidence, 6),
            }
            for rank, (token, distance) in enumerate(recognition.hypotheses[:5], start=1):
                row[f"top{rank}"] = f"{token}:{distance:.6f}"
            writer.writerow(row)


def save_hypotheses(recognitions: list[Recognition]) -> None:
    lines: list[str] = []
    for recognition in recognitions:
        hypotheses = ", ".join(f"{token}={distance:.4f}" for token, distance in recognition.hypotheses)
        lines.append(
            f"{recognition.index}: expected={recognition.expected}, recognized={recognition.recognized}, "
            f"hypotheses=[{hypotheses}]"
        )
    HYPOTHESES_PATH.write_text("\n".join(lines), encoding="utf-8")


def stft_power(signal: np.ndarray, sample_rate: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    window = np.hanning(SPECTROGRAM_WINDOW_SIZE)
    frame_count = int(np.ceil(max(signal.size - SPECTROGRAM_WINDOW_SIZE, 0) / SPECTROGRAM_HOP_SIZE)) + 1
    padded_length = (frame_count - 1) * SPECTROGRAM_HOP_SIZE + SPECTROGRAM_WINDOW_SIZE
    padded = np.pad(signal, (0, max(0, padded_length - signal.size)))
    power = np.empty((SPECTROGRAM_N_FFT // 2 + 1, frame_count), dtype=np.float64)

    for index in range(frame_count):
        start = index * SPECTROGRAM_HOP_SIZE
        frame = padded[start : start + SPECTROGRAM_WINDOW_SIZE] * window
        spectrum = np.fft.rfft(frame, n=SPECTROGRAM_N_FFT)
        power[:, index] = np.abs(spectrum) ** 2

    frequencies = np.fft.rfftfreq(SPECTROGRAM_N_FFT, d=1.0 / sample_rate)
    times = (np.arange(frame_count) * SPECTROGRAM_HOP_SIZE + SPECTROGRAM_WINDOW_SIZE / 2) / sample_rate
    return power, frequencies, times


def save_phone_spectrogram(signal: np.ndarray, sample_rate: int, path: Path) -> None:
    power, frequencies, times = stft_power(signal, sample_rate)
    mask = (frequencies >= 50) & (frequencies <= 8000)
    normalized = power[mask] / max(float(power[mask].max()), 1e-12)
    db = 10.0 * np.log10(normalized + 1e-12)

    fig, ax = plt.subplots(figsize=(11, 5))
    mesh = ax.pcolormesh(times, frequencies[mask], db, shading="auto", cmap="magma", vmin=-90, vmax=0)
    ax.set_yscale("log")
    ax.set_ylim(50, 8000)
    ax.set_title("Спектрограмма записи телефонного номера")
    ax.set_xlabel("Время, с")
    ax.set_ylabel("Частота, Гц")
    ax.set_yticks([50, 100, 200, 500, 1000, 2000, 5000, 8000])
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    fig.colorbar(mesh, ax=ax, label="Уровень, дБ от максимума")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_segmentation_preview(
    signal: np.ndarray,
    sample_rate: int,
    segments: list[Segment],
    threshold: float,
    path: Path,
) -> None:
    values, starts = frame_rms(signal, sample_rate, FRAME_SECONDS, HOP_SECONDS)
    times = np.arange(signal.size) / sample_rate
    frame_times = starts / sample_rate

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    axes[0].plot(times, signal, color="black", linewidth=0.55)
    axes[0].set_title("Сегментация телефонной записи")
    axes[0].set_ylabel("Амплитуда")
    axes[0].grid(axis="both", linestyle=":", alpha=0.35)

    for segment in segments:
        start_s = segment.start_sample / sample_rate
        end_s = segment.end_sample / sample_rate
        axes[0].axvspan(start_s, end_s, color="tab:green", alpha=0.18)
        axes[0].text(
            (start_s + end_s) / 2,
            0.92,
            str(segment.index),
            ha="center",
            va="top",
            transform=axes[0].get_xaxis_transform(),
            fontsize=9,
        )

    axes[1].plot(frame_times, values, color="black", linewidth=0.8)
    axes[1].axhline(threshold, color="crimson", linestyle="--", linewidth=1.0, label=f"порог = {threshold:.5f}")
    axes[1].set_xlabel("Время, с")
    axes[1].set_ylabel("RMS")
    axes[1].grid(axis="both", linestyle=":", alpha=0.35)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_classification_preview(segments: list[Segment], recognitions: list[Recognition], sample_rate: int, path: Path) -> None:
    cols = 4
    rows = int(math.ceil(len(segments) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 2.3))
    axes = np.atleast_1d(axes).ravel()

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor("#f2f2f2")

    for ax, segment, recognition in zip(axes, segments, recognitions, strict=False):
        times = np.arange(segment.signal.size) / sample_rate
        ax.plot(times, segment.signal, color="black", linewidth=0.7)
        color = "green" if recognition.expected == recognition.recognized else "crimson"
        ax.set_title(
            f"{segment.index}: {token_label(recognition.expected)} -> {token_label(recognition.recognized)}",
            color=color,
            fontsize=10,
        )
        ax.set_ylim(-1.05, 1.05)

    for ax in axes[len(segments) :]:
        ax.axis("off")

    fig.suptitle("Результат распознавания сегментов", fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_metrics_csv(
    phone_info: WavInfo,
    truth: list[str],
    segments: list[Segment],
    recognitions: list[Recognition],
    threshold: float,
) -> tuple[int, float]:
    recognized = [recognition.recognized for recognition in recognitions]
    compared_count = min(len(truth), len(recognized))
    substitutions = sum(truth[index] != recognized[index] for index in range(compared_count))
    length_errors = abs(len(truth) - len(recognized))
    errors = substitutions + length_errors
    accuracy = 100.0 * (len(truth) - errors) / len(truth) if truth else 0.0
    confidence_values = [recognition.confidence for recognition in recognitions]
    mean_confidence = 100.0 * float(np.mean(confidence_values)) if confidence_values else 0.0
    min_confidence = 100.0 * float(np.min(confidence_values)) if confidence_values else 0.0

    rows: list[dict[str, str | int | float]] = [
        {"metric": "variant", "value": VARIANT, "unit": ""},
        {"metric": "lab_variant", "value": LAB_VARIANT, "unit": ""},
        {"metric": "sample_rate", "value": phone_info.sample_rate, "unit": "Hz"},
        {"metric": "channels", "value": phone_info.channels, "unit": ""},
        {"metric": "phone_duration", "value": round(phone_info.duration, 3), "unit": "s"},
        {"metric": "alphabet_size", "value": len(TOKENS), "unit": "files"},
        {"metric": "truth_count", "value": len(truth), "unit": "tokens"},
        {"metric": "segment_count", "value": len(segments), "unit": "segments"},
        {"metric": "segmentation_threshold", "value": round(threshold, 8), "unit": "RMS"},
        {"metric": "recognized_sequence", "value": " ".join(recognized), "unit": ""},
        {"metric": "expected_sequence", "value": " ".join(truth), "unit": ""},
        {"metric": "errors", "value": errors, "unit": "tokens"},
        {"metric": "accuracy", "value": round(accuracy, 3), "unit": "%"},
        {"metric": "mean_confidence", "value": round(mean_confidence, 3), "unit": "%"},
        {"metric": "min_confidence", "value": round(min_confidence, 3), "unit": "%"},
        {"metric": "feature_frame", "value": int(FEATURE_FRAME_SECONDS * phone_info.sample_rate), "unit": "samples"},
        {"metric": "feature_hop", "value": int(FEATURE_HOP_SECONDS * phone_info.sample_rate), "unit": "samples"},
        {"metric": "n_fft", "value": N_FFT, "unit": "samples"},
        {"metric": "mel_bands", "value": MEL_BANDS, "unit": ""},
        {"metric": "mfcc_count", "value": MFCC_COUNT - 1, "unit": "coefficients"},
    ]

    with METRICS_CSV_PATH.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "value", "unit"], delimiter=";")
        writer.writeheader()
        writer.writerows(rows)

    return errors, accuracy


def run_pipeline() -> dict[str, str | int | float]:
    ensure_dirs([SEGMENT_DIR, SPECTROGRAM_DIR, PREVIEW_DIR])
    for directory, pattern in [
        (SEGMENT_DIR, "*.wav"),
        (SPECTROGRAM_DIR, "*.png"),
        (PREVIEW_DIR, "*.png"),
    ]:
        clear_files(directory, pattern)

    truth = read_truth()
    phone_signal, phone_info = read_wav_mono(PHONE_PATH)
    alphabet_features, alphabet_rows = load_alphabet()
    save_alphabet_csv(alphabet_rows)

    segment_bounds, threshold = segment_phone(phone_signal, phone_info.sample_rate, truth)
    segments = save_segments(phone_signal, phone_info.sample_rate, segment_bounds, truth)
    recognitions = recognize_segments(segments, alphabet_features, phone_info.sample_rate)

    save_segments_csv(segments, phone_info.sample_rate)
    save_recognition_csv(recognitions)
    save_hypotheses(recognitions)
    save_phone_spectrogram(phone_signal, phone_info.sample_rate, SPECTROGRAM_DIR / "phone_spectrogram.png")
    save_segmentation_preview(phone_signal, phone_info.sample_rate, segments, threshold, PREVIEW_DIR / "segmentation_overview.png")
    save_classification_preview(segments, recognitions, phone_info.sample_rate, PREVIEW_DIR / "classification_overview.png")
    errors, accuracy = save_metrics_csv(phone_info, truth, segments, recognitions, threshold)

    recognized = " ".join(recognition.recognized for recognition in recognitions)
    return {
        "variant": VARIANT,
        "lab_variant": LAB_VARIANT,
        "sample_rate": phone_info.sample_rate,
        "phone_duration": round(phone_info.duration, 3),
        "truth": " ".join(truth),
        "recognized": recognized,
        "segment_count": len(segments),
        "errors": errors,
        "accuracy_percent": round(accuracy, 3),
        "mean_confidence_percent": round(
            100.0 * float(np.mean([recognition.confidence for recognition in recognitions])), 3
        )
        if recognitions
        else 0.0,
        "recognition_csv": str(RECOGNITION_CSV_PATH),
        "segments_csv": str(SEGMENTS_CSV_PATH),
        "metrics_csv": str(METRICS_CSV_PATH),
        "phone_spectrogram": str(SPECTROGRAM_DIR / "phone_spectrogram.png"),
    }


if __name__ == "__main__":
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    summary = run_pipeline()
    for key, value in summary.items():
        print(f"{key}: {value}")
