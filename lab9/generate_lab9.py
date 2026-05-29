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
INSTRUMENT = "синтезированное фортепиано"
NOTE_NAME = "A4 / Ля первой октавы"
EXPECTED_FREQUENCY_HZ = 440.0

WINDOW_NAME = "Hann"
WINDOW_SIZE = 4096
HOP_SIZE = 1024
N_FFT = 8192
NOISE_SUBTRACTION_STRENGTH = 1.35
SPECTRAL_FLOOR = 0.04
QUIET_EDGE_SECONDS = 1.5
ENERGY_DT_SECONDS = 0.1
ENERGY_DF_HZ = 50
ENERGY_MIN_FREQ_HZ = 50
ENERGY_MAX_FREQ_HZ = 5000

BASE_DIR = Path(__file__).resolve().parent
AUDIO_PATH = BASE_DIR / "piano_A4_mono_48k_16bit_pcm.wav"
NOISE_PATH = BASE_DIR / "room_noise_only_mono_48k_16bit_pcm.wav"

SPECTROGRAM_DIR = BASE_DIR / "spectrogram_png"
WAVEFORM_DIR = BASE_DIR / "waveform_png"
PREVIEW_DIR = BASE_DIR / "preview_png"
DENOISED_DIR = BASE_DIR / "denoised_wav"

DENOISED_WAV_PATH = DENOISED_DIR / "piano_A4_denoised_variant17.wav"
METRICS_PATH = BASE_DIR / "metrics_variant17.csv"
ENERGY_PEAKS_PATH = BASE_DIR / "energy_peaks_variant17.csv"


@dataclass
class WavInfo:
    path: Path
    sample_rate: int
    channels: int
    sample_width: int
    frames: int
    duration: float


@dataclass
class StftResult:
    spectrum: np.ndarray
    power: np.ndarray
    frequencies: np.ndarray
    times: np.ndarray


def ensure_dirs(paths: Iterable[Path]) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def clear_files(path: Path, pattern: str) -> None:
    for file in path.glob(pattern):
        file.unlink()


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


def amplitude_db(value: float, reference: float = 1.0) -> float:
    return 20.0 * math.log10(max(value, 1e-12) / max(reference, 1e-12))


def ratio_db(numerator: float, denominator: float) -> float:
    return 20.0 * math.log10(max(numerator, 1e-12) / max(denominator, 1e-12))


def make_window() -> np.ndarray:
    return np.hanning(WINDOW_SIZE).astype(np.float64)


def stft(signal: np.ndarray) -> StftResult:
    window = make_window()
    left_pad = WINDOW_SIZE // 2
    centered = np.pad(signal, (left_pad, 0))
    if centered.size <= WINDOW_SIZE:
        frame_count = 1
    else:
        frame_count = int(np.ceil((centered.size - WINDOW_SIZE) / HOP_SIZE)) + 1

    padded_length = (frame_count - 1) * HOP_SIZE + WINDOW_SIZE
    padded = np.pad(centered, (0, padded_length - centered.size))

    frames = np.empty((frame_count, N_FFT // 2 + 1), dtype=np.complex128)
    for index in range(frame_count):
        start = index * HOP_SIZE
        frame = padded[start : start + WINDOW_SIZE] * window
        frames[index] = np.fft.rfft(frame, n=N_FFT)

    spectrum = frames.T
    power = np.abs(spectrum) ** 2
    frequencies = np.fft.rfftfreq(N_FFT, d=1.0)
    times = (np.arange(frame_count) * HOP_SIZE + WINDOW_SIZE / 2 - left_pad) / 1.0
    return StftResult(spectrum=spectrum, power=power, frequencies=frequencies, times=times)


def stft_with_rate(signal: np.ndarray, sample_rate: int) -> StftResult:
    result = stft(signal)
    return StftResult(
        spectrum=result.spectrum,
        power=result.power,
        frequencies=result.frequencies * sample_rate,
        times=result.times / sample_rate,
    )


def istft(spectrum: np.ndarray, output_length: int) -> np.ndarray:
    window = make_window()
    left_pad = WINDOW_SIZE // 2
    frame_count = spectrum.shape[1]
    reconstructed_length = (frame_count - 1) * HOP_SIZE + WINDOW_SIZE
    output = np.zeros(reconstructed_length, dtype=np.float64)
    window_sum = np.zeros(reconstructed_length, dtype=np.float64)

    for index in range(frame_count):
        frame = np.fft.irfft(spectrum[:, index], n=N_FFT)[:WINDOW_SIZE]
        start = index * HOP_SIZE
        output[start : start + WINDOW_SIZE] += frame * window
        window_sum[start : start + WINDOW_SIZE] += window**2

    active = window_sum > 1e-10
    output[active] /= window_sum[active]
    return output[left_pad : left_pad + output_length]


def denoise_by_spectral_subtraction(signal: np.ndarray, noise: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    signal_stft = stft(signal)
    noise_stft = stft(noise)

    magnitude = np.abs(signal_stft.spectrum)
    noise_profile = np.median(np.abs(noise_stft.spectrum), axis=1)
    subtracted = magnitude - NOISE_SUBTRACTION_STRENGTH * noise_profile[:, None]
    denoised_magnitude = np.maximum(subtracted, SPECTRAL_FLOOR * magnitude)
    gain = denoised_magnitude / (magnitude + 1e-12)
    gain = np.clip(gain, 0.0, 1.0)
    denoised_spectrum = signal_stft.spectrum * gain
    denoised = istft(denoised_spectrum, signal.size)
    return denoised, noise_profile


def edge_noise(signal: np.ndarray, sample_rate: int) -> np.ndarray:
    edge_samples = int(QUIET_EDGE_SECONDS * sample_rate)
    if signal.size <= edge_samples * 2:
        return signal
    return np.concatenate([signal[:edge_samples], signal[-edge_samples:]])


def detect_active_region(signal: np.ndarray, noise: np.ndarray, sample_rate: int) -> tuple[float, float]:
    frame_size = int(0.05 * sample_rate)
    hop = frame_size
    noise_rms = rms(noise)
    frame_rms: list[float] = []
    starts: list[int] = []

    for start in range(0, max(signal.size - frame_size + 1, 1), hop):
        frame = signal[start : start + frame_size]
        if frame.size < frame_size:
            frame = np.pad(frame, (0, frame_size - frame.size))
        starts.append(start)
        frame_rms.append(rms(frame))

    values = np.array(frame_rms)
    threshold = max(noise_rms * 2.0, float(np.percentile(values, 75)) * 0.03)
    active = values > threshold
    if not np.any(active):
        return 0.0, signal.size / sample_rate

    first = int(np.argmax(active))
    last = int(len(active) - np.argmax(active[::-1]) - 1)
    start_sec = max(0.0, (starts[first] - frame_size) / sample_rate)
    end_sec = min(signal.size / sample_rate, (starts[last] + frame_size * 2) / sample_rate)
    return start_sec, end_sec


def save_spectrogram(title: str, stft_result: StftResult, path: Path, max_freq: int = 12000) -> None:
    frequencies = stft_result.frequencies
    times = stft_result.times
    mask = (frequencies >= 20) & (frequencies <= max_freq)
    power = stft_result.power[mask]
    normalized = power / max(float(power.max()), 1e-12)
    db = 10.0 * np.log10(normalized + 1e-12)

    fig, ax = plt.subplots(figsize=(11, 5))
    mesh = ax.pcolormesh(times, frequencies[mask], db, shading="auto", cmap="magma", vmin=-90, vmax=0)
    ax.set_yscale("log")
    ax.set_ylim(20, max_freq)
    ax.set_title(title)
    ax.set_xlabel("Время, с")
    ax.set_ylabel("Частота, Гц")
    ticks = [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
    ax.set_yticks([tick for tick in ticks if tick <= max_freq])
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    fig.colorbar(mesh, ax=ax, label="Уровень, дБ от максимума")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_waveform_compare(original: np.ndarray, denoised: np.ndarray, sample_rate: int, path: Path) -> None:
    times = np.arange(original.size) / sample_rate
    fig, axes = plt.subplots(2, 1, figsize=(11, 5), sharex=True)
    axes[0].plot(times, original, color="black", linewidth=0.7)
    axes[0].set_title("Исходный сигнал")
    axes[1].plot(times, denoised, color="black", linewidth=0.7)
    axes[1].set_title("После вычитания шума")
    for ax in axes:
        ax.set_ylabel("Амплитуда")
        ax.grid(axis="both", linestyle=":", alpha=0.35)
        ax.set_ylim(-1.05, 1.05)
    axes[1].set_xlabel("Время, с")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_noise_profile(
    frequencies: np.ndarray,
    source_spectrum: np.ndarray,
    noise_profile: np.ndarray,
    active_region: tuple[float, float],
    times: np.ndarray,
    path: Path,
) -> None:
    start_sec, end_sec = active_region
    frame_mask = (times >= start_sec) & (times <= end_sec)
    if not np.any(frame_mask):
        frame_mask = np.ones_like(times, dtype=bool)
    source_profile = np.mean(np.abs(source_spectrum[:, frame_mask]), axis=1)
    freq_mask = (frequencies >= 20) & (frequencies <= 12000)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(
        frequencies[freq_mask],
        20.0 * np.log10(source_profile[freq_mask] + 1e-12),
        label="Сигнал, средний спектр активной части",
        color="black",
        linewidth=1.1,
    )
    ax.plot(
        frequencies[freq_mask],
        20.0 * np.log10(noise_profile[freq_mask] + 1e-12),
        label="Профиль шума",
        color="crimson",
        linewidth=1.1,
    )
    ax.set_xscale("log")
    ax.set_title("Оценка спектрального профиля шума")
    ax.set_xlabel("Частота, Гц")
    ax.set_ylabel("Амплитуда, дБ")
    ax.grid(axis="both", linestyle=":", alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def find_peak_frequency(stft_result: StftResult, active_region: tuple[float, float], low: float, high: float) -> float:
    start_sec, end_sec = active_region
    frame_mask = (stft_result.times >= start_sec) & (stft_result.times <= end_sec)
    freq_mask = (stft_result.frequencies >= low) & (stft_result.frequencies <= high)
    if not np.any(frame_mask):
        frame_mask = np.ones_like(stft_result.times, dtype=bool)
    average_power = np.mean(stft_result.power[np.ix_(freq_mask, frame_mask)], axis=1)
    selected_freqs = stft_result.frequencies[freq_mask]
    return float(selected_freqs[int(np.argmax(average_power))])


def aggregate_energy_windows(stft_result: StftResult, duration: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    time_edges = np.arange(0.0, duration + ENERGY_DT_SECONDS, ENERGY_DT_SECONDS)
    freq_edges = np.arange(ENERGY_MIN_FREQ_HZ, ENERGY_MAX_FREQ_HZ + ENERGY_DF_HZ, ENERGY_DF_HZ)
    grid = np.zeros((freq_edges.size - 1, time_edges.size - 1), dtype=np.float64)

    for time_index in range(time_edges.size - 1):
        time_mask = (stft_result.times >= time_edges[time_index]) & (stft_result.times < time_edges[time_index + 1])
        if not np.any(time_mask):
            nearest = int(np.argmin(np.abs(stft_result.times - (time_edges[time_index] + ENERGY_DT_SECONDS / 2))))
            time_mask[nearest] = True

        for freq_index in range(freq_edges.size - 1):
            freq_mask = (stft_result.frequencies >= freq_edges[freq_index]) & (
                stft_result.frequencies < freq_edges[freq_index + 1]
            )
            grid[freq_index, time_index] = float(stft_result.power[np.ix_(freq_mask, time_mask)].sum())

    return grid, time_edges, freq_edges


def top_energy_rows(grid: np.ndarray, time_edges: np.ndarray, freq_edges: np.ndarray, limit: int = 12) -> list[dict[str, float | int]]:
    flat_order = np.argsort(grid.ravel())[::-1][:limit]
    max_energy = float(grid.max())
    rows: list[dict[str, float | int]] = []
    for rank, flat_index in enumerate(flat_order, start=1):
        freq_index, time_index = np.unravel_index(flat_index, grid.shape)
        energy = float(grid[freq_index, time_index])
        rows.append(
            {
                "rank": rank,
                "time_start_s": round(float(time_edges[time_index]), 3),
                "time_end_s": round(float(time_edges[time_index + 1]), 3),
                "freq_start_hz": round(float(freq_edges[freq_index]), 1),
                "freq_end_hz": round(float(freq_edges[freq_index + 1]), 1),
                "time_center_s": round(float((time_edges[time_index] + time_edges[time_index + 1]) / 2), 3),
                "freq_center_hz": round(float((freq_edges[freq_index] + freq_edges[freq_index + 1]) / 2), 1),
                "energy": round(energy, 6),
                "energy_db_relative": round(10.0 * math.log10(max(energy, 1e-12) / max(max_energy, 1e-12)), 3),
            }
        )
    return rows


def save_energy_peaks_csv(rows: list[dict[str, float | int]], path: Path) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter=";")
        writer.writeheader()
        writer.writerows(rows)


def save_energy_map(
    grid: np.ndarray,
    time_edges: np.ndarray,
    freq_edges: np.ndarray,
    top_rows: list[dict[str, float | int]],
    path: Path,
) -> None:
    normalized = grid / max(float(grid.max()), 1e-12)
    db = 10.0 * np.log10(normalized + 1e-12)
    fig, ax = plt.subplots(figsize=(11, 5))
    mesh = ax.pcolormesh(time_edges, freq_edges, db, shading="auto", cmap="viridis", vmin=-70, vmax=0)
    ax.set_yscale("log")
    ax.set_ylim(ENERGY_MIN_FREQ_HZ, ENERGY_MAX_FREQ_HZ)
    ax.set_title("Энергия в окнах 0.1 с × 50 Гц")
    ax.set_xlabel("Время, с")
    ax.set_ylabel("Частота, Гц")
    ax.set_yticks([50, 100, 200, 500, 1000, 2000, 5000])
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    for row in top_rows[:8]:
        ax.scatter(row["time_center_s"], row["freq_center_hz"], color="white", s=22, edgecolor="black", linewidth=0.4)
    fig.colorbar(mesh, ax=ax, label="Энергия, дБ от максимума")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_overview(
    original: np.ndarray,
    denoised: np.ndarray,
    original_stft: StftResult,
    denoised_stft: StftResult,
    sample_rate: int,
    path: Path,
) -> None:
    times = np.arange(original.size) / sample_rate
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    axes[0, 0].plot(times, original, color="black", linewidth=0.6)
    axes[0, 0].set_title("Исходный сигнал")
    axes[1, 0].plot(times, denoised, color="black", linewidth=0.6)
    axes[1, 0].set_title("После вычитания шума")

    for ax, stft_result, title in [
        (axes[0, 1], original_stft, "Спектрограмма до"),
        (axes[1, 1], denoised_stft, "Спектрограмма после"),
    ]:
        freq_mask = (stft_result.frequencies >= 20) & (stft_result.frequencies <= 12000)
        power = stft_result.power[freq_mask]
        db = 10.0 * np.log10(power / max(float(power.max()), 1e-12) + 1e-12)
        ax.pcolormesh(stft_result.times, stft_result.frequencies[freq_mask], db, shading="auto", cmap="magma", vmin=-90, vmax=0)
        ax.set_yscale("log")
        ax.set_ylim(20, 12000)
        ax.set_title(title)
        ax.set_ylabel("Частота, Гц")
        ax.set_xlabel("Время, с")

    for ax in axes[:, 0]:
        ax.set_xlabel("Время, с")
        ax.set_ylabel("Амплитуда")
        ax.grid(axis="both", linestyle=":", alpha=0.35)
        ax.set_ylim(-1.05, 1.05)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_metrics_csv(rows: list[dict[str, str | int | float]], path: Path) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "value", "unit"], delimiter=";")
        writer.writeheader()
        writer.writerows(rows)


def run_pipeline() -> dict[str, str | int | float]:
    ensure_dirs([SPECTROGRAM_DIR, WAVEFORM_DIR, PREVIEW_DIR, DENOISED_DIR])
    for directory in [SPECTROGRAM_DIR, WAVEFORM_DIR, PREVIEW_DIR]:
        clear_files(directory, "*.png")
    clear_files(DENOISED_DIR, "*.wav")

    signal, audio_info = read_wav_mono(AUDIO_PATH)
    noise, noise_info = read_wav_mono(NOISE_PATH)
    if audio_info.sample_rate != noise_info.sample_rate:
        raise ValueError("Частота дискретизации сигнала и шума должна совпадать")

    sample_rate = audio_info.sample_rate
    denoised, noise_profile = denoise_by_spectral_subtraction(signal, noise)
    write_wav_16bit(DENOISED_WAV_PATH, denoised, sample_rate)

    original_stft = stft_with_rate(signal, sample_rate)
    denoised_stft = stft_with_rate(denoised, sample_rate)
    active_region = detect_active_region(signal, noise, sample_rate)

    save_spectrogram(
        "Спектрограмма исходного сигнала",
        original_stft,
        SPECTROGRAM_DIR / "original_spectrogram.png",
    )
    save_spectrogram(
        "Спектрограмма после вычитания шума",
        denoised_stft,
        SPECTROGRAM_DIR / "denoised_spectrogram.png",
    )
    save_waveform_compare(signal, denoised, sample_rate, WAVEFORM_DIR / "waveform_compare.png")
    save_noise_profile(
        original_stft.frequencies,
        original_stft.spectrum,
        noise_profile,
        active_region,
        original_stft.times,
        WAVEFORM_DIR / "noise_profile.png",
    )

    grid, time_edges, freq_edges = aggregate_energy_windows(original_stft, audio_info.duration)
    top_rows = top_energy_rows(grid, time_edges, freq_edges)
    save_energy_peaks_csv(top_rows, ENERGY_PEAKS_PATH)
    save_energy_map(grid, time_edges, freq_edges, top_rows, PREVIEW_DIR / "energy_windows.png")
    save_overview(signal, denoised, original_stft, denoised_stft, sample_rate, PREVIEW_DIR / "analysis_overview.png")

    start_sec, end_sec = active_region
    start_index = int(start_sec * sample_rate)
    end_index = int(end_sec * sample_rate)
    background_before = edge_noise(signal, sample_rate)
    background_after = edge_noise(denoised, sample_rate)
    active_before = signal[start_index:end_index]
    active_after = denoised[start_index:end_index]

    note_peak_hz = find_peak_frequency(original_stft, active_region, 300.0, 600.0)
    dominant_hz = find_peak_frequency(original_stft, active_region, 50.0, 5000.0)

    background_rms_before = rms(background_before)
    background_rms_after = rms(background_after)
    active_rms_before = rms(active_before)
    active_rms_after = rms(active_after)
    snr_before = ratio_db(active_rms_before, background_rms_before)
    snr_after = ratio_db(active_rms_after, background_rms_after)

    metrics_rows: list[dict[str, str | int | float]] = [
        {"metric": "variant", "value": VARIANT, "unit": ""},
        {"metric": "instrument", "value": INSTRUMENT, "unit": ""},
        {"metric": "note", "value": NOTE_NAME, "unit": ""},
        {"metric": "sample_rate", "value": sample_rate, "unit": "Hz"},
        {"metric": "channels", "value": audio_info.channels, "unit": ""},
        {"metric": "duration", "value": round(audio_info.duration, 3), "unit": "s"},
        {"metric": "window", "value": WINDOW_NAME, "unit": ""},
        {"metric": "window_size", "value": WINDOW_SIZE, "unit": "samples"},
        {"metric": "hop_size", "value": HOP_SIZE, "unit": "samples"},
        {"metric": "n_fft", "value": N_FFT, "unit": "samples"},
        {"metric": "frequency_step", "value": round(sample_rate / N_FFT, 3), "unit": "Hz"},
        {"metric": "active_start", "value": round(start_sec, 3), "unit": "s"},
        {"metric": "active_end", "value": round(end_sec, 3), "unit": "s"},
        {"metric": "noise_file_rms", "value": round(rms(noise), 8), "unit": "amplitude"},
        {"metric": "background_rms_before", "value": round(background_rms_before, 8), "unit": "amplitude"},
        {"metric": "background_rms_after", "value": round(background_rms_after, 8), "unit": "amplitude"},
        {"metric": "active_rms_before", "value": round(active_rms_before, 8), "unit": "amplitude"},
        {"metric": "active_rms_after", "value": round(active_rms_after, 8), "unit": "amplitude"},
        {"metric": "background_reduction", "value": round(ratio_db(background_rms_before, background_rms_after), 3), "unit": "dB"},
        {"metric": "snr_before", "value": round(snr_before, 3), "unit": "dB"},
        {"metric": "snr_after", "value": round(snr_after, 3), "unit": "dB"},
        {"metric": "snr_gain", "value": round(snr_after - snr_before, 3), "unit": "dB"},
        {"metric": "note_peak_frequency", "value": round(note_peak_hz, 3), "unit": "Hz"},
        {"metric": "dominant_frequency", "value": round(dominant_hz, 3), "unit": "Hz"},
        {"metric": "expected_frequency", "value": EXPECTED_FREQUENCY_HZ, "unit": "Hz"},
        {"metric": "note_frequency_error", "value": round(note_peak_hz - EXPECTED_FREQUENCY_HZ, 3), "unit": "Hz"},
        {"metric": "top_energy_time", "value": top_rows[0]["time_center_s"], "unit": "s"},
        {"metric": "top_energy_frequency", "value": top_rows[0]["freq_center_hz"], "unit": "Hz"},
    ]
    save_metrics_csv(metrics_rows, METRICS_PATH)

    return {
        "variant": VARIANT,
        "instrument": INSTRUMENT,
        "note": NOTE_NAME,
        "sample_rate": sample_rate,
        "duration": round(audio_info.duration, 3),
        "active_region": f"{start_sec:.3f}-{end_sec:.3f} s",
        "background_reduction_db": round(ratio_db(background_rms_before, background_rms_after), 3),
        "snr_before_db": round(snr_before, 3),
        "snr_after_db": round(snr_after, 3),
        "note_peak_frequency_hz": round(note_peak_hz, 3),
        "dominant_frequency_hz": round(dominant_hz, 3),
        "denoised_wav": str(DENOISED_WAV_PATH),
        "metrics_csv": str(METRICS_PATH),
        "energy_peaks_csv": str(ENERGY_PEAKS_PATH),
        "overview": str(PREVIEW_DIR / "analysis_overview.png"),
    }


if __name__ == "__main__":
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    summary = run_pipeline()
    for key, value in summary.items():
        print(f"{key}: {value}")
