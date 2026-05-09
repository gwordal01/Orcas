"""
EchoKiller / EchoK
Adaptive FIR echo attenuation demo with a more polished dashboard.

What this file demonstrates:
- load a WAV file if one exists
- otherwise generate a personalized synthetic source
- add a controllable synthetic echo path
- learn that echo path with an adaptive FIR filter (NLMS-style LMS)
- save before/after audio
- display waveforms and learned coefficients in a sleeker layout
"""

from __future__ import annotations

import argparse
import os
import time
import wave

import matplotlib.pyplot as plt
import numpy as np

try:
    from scipy import signal

    HAS_SCIPY = True
except ImportError:
    signal = None
    HAS_SCIPY = False

try:
    import soundfile as sf

    HAS_SOUNDFILE = True
except ImportError:
    HAS_SOUNDFILE = False

try:
    import sounddevice as sd

    HAS_PLAYBACK = True
except ImportError:
    HAS_PLAYBACK = False


SAMPLE_RATE = 16000
PROFILE_NAME = "Gwordal"
PROFILE_TAGLINE = "AEC demo tuned for clear before/after contrast"
ECHO_DELAY_MS = 120
FILTER_ORDER = 2048
LEARNING_RATE = 0.035
OUTPUT_CLEAN = "cleaned_output.wav"
OUTPUT_ECHOED = "echoed_input.wav"
OUTPUT_REFERENCE = "reference_input.wav"
OUTPUT_FIGURE = "echok_dashboard.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Adaptive FIR EchoKiller demo")
    parser.add_argument("--name", default=PROFILE_NAME, help="Name shown in the dashboard")
    parser.add_argument("--no-playback", action="store_true", help="Skip audio playback")
    parser.add_argument("--no-show", action="store_true", help="Save plots without opening a window")
    return parser.parse_args()


def normalize_audio(x: np.ndarray, peak: float = 0.92) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    max_abs = float(np.max(np.abs(x))) if x.size else 0.0
    if max_abs < 1e-8:
        return x.copy()
    return (x / max_abs * peak).astype(np.float32)


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x), dtype=np.float64)))


def generate_personalized_source(name: str, duration: float = 4.0, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """
    Build a source that is easier to demo than a plain tone:
    speech-like bursts + a soft musical bed for a more obvious audible difference.
    """
    t = np.linspace(0, duration, int(duration * sample_rate), endpoint=False, dtype=np.float32)
    source = np.zeros_like(t)

    bursts = [
        (0.20, 0.45, 155.0),
        (0.95, 0.40, 182.0),
        (1.55, 0.55, 130.0),
        (2.40, 0.50, 205.0),
        (3.10, 0.45, 175.0),
    ]

    for start, length, fundamental in bursts:
        mask = (t >= start) & (t < start + length)
        if not np.any(mask):
            continue
        local_t = t[mask] - start
        envelope = np.sin(np.pi * local_t / length) ** 2
        voice = np.zeros(np.count_nonzero(mask), dtype=np.float32)
        for harmonic in range(1, 5):
            voice += np.sin(2 * np.pi * fundamental * harmonic * local_t) / harmonic
        tremolo = 1.0 + 0.12 * np.sin(2 * np.pi * 5.0 * local_t)
        source[mask] += voice * envelope * tremolo

    seed = sum(ord(ch) for ch in name) % 12
    root_hz = 220.0 * (2 ** ((seed - 9) / 12.0))
    chord = [root_hz, root_hz * 5 / 4, root_hz * 3 / 2]
    bed = np.zeros_like(t)
    for idx, freq in enumerate(chord):
        bed += 0.08 * np.sin(2 * np.pi * freq * t + idx * 0.6)
    bed *= 0.5 + 0.5 * np.sin(2 * np.pi * 0.35 * t) ** 2

    combined = source + bed
    if HAS_SCIPY:
        combined = signal.lfilter([1.0], [1.0, -0.82], combined)
    else:
        smooth = np.empty_like(combined)
        prev = 0.0
        for idx, sample in enumerate(combined):
            prev = sample + 0.82 * prev
            smooth[idx] = prev
        combined = smooth
    return normalize_audio(combined)


def add_synthetic_echo(x: np.ndarray, delay_ms: int = ECHO_DELAY_MS, sample_rate: int = SAMPLE_RATE) -> tuple[np.ndarray, np.ndarray]:
    """
    Create a small multi-path echo path to mimic room reflections.
    The learned FIR taps should resemble this impulse response.
    """
    delay = int(sample_rate * delay_ms / 1000)
    taps = np.zeros(delay + 220, dtype=np.float32)
    taps[0] = 1.0
    taps[delay] = 0.58
    taps[min(delay + 75, len(taps) - 1)] = 0.24
    taps[min(delay + 140, len(taps) - 1)] = -0.12
    if HAS_SCIPY:
        echoed = signal.lfilter(taps, [1.0], x).astype(np.float32)
    else:
        echoed = np.convolve(x, taps, mode="full")[: len(x)].astype(np.float32)
    return normalize_audio(echoed), taps


def load_reference_audio() -> tuple[np.ndarray, bool, str]:
    wav_files = sorted(
        f
        for f in os.listdir(".")
        if f.lower().endswith(".wav") and f not in {OUTPUT_CLEAN, OUTPUT_ECHOED, OUTPUT_REFERENCE}
    )

    if wav_files and HAS_SOUNDFILE:
        path = wav_files[0]
        data, sr = sf.read(path)
        if data.ndim > 1:
            data = data[:, 0]
        data = np.asarray(data, dtype=np.float32)
        if sr != SAMPLE_RATE:
            if HAS_SCIPY:
                gcd = np.gcd(sr, SAMPLE_RATE)
                data = signal.resample_poly(data, SAMPLE_RATE // gcd, sr // gcd).astype(np.float32)
            else:
                old_idx = np.linspace(0.0, len(data) - 1, num=len(data), dtype=np.float32)
                new_len = int(len(data) * SAMPLE_RATE / sr)
                new_idx = np.linspace(0.0, len(data) - 1, num=new_len, dtype=np.float32)
                data = np.interp(new_idx, old_idx, data).astype(np.float32)
        return normalize_audio(data), True, path

    return generate_personalized_source(PROFILE_NAME), False, "synthetic"


def lms_filter(reference: np.ndarray, mixed: np.ndarray, filter_order: int, mu: float) -> tuple[np.ndarray, np.ndarray]:
    n_samples = min(len(reference), len(mixed))
    reference = reference[:n_samples].astype(np.float32, copy=False)
    mixed = mixed[:n_samples].astype(np.float32, copy=False)

    weights = np.zeros(filter_order, dtype=np.float32)
    error = np.zeros(n_samples, dtype=np.float32)
    ref_buffer = np.zeros(filter_order, dtype=np.float32)

    for n in range(n_samples):
        ref_buffer[1:] = ref_buffer[:-1]
        ref_buffer[0] = reference[n]
        predicted = float(np.dot(weights, ref_buffer))
        err = mixed[n] - predicted
        error[n] = err
        power = float(np.dot(ref_buffer, ref_buffer)) + 1e-6
        weights += (mu / power) * err * ref_buffer

    return normalize_audio(error), weights


def style_axes(ax: plt.Axes) -> None:
    ax.set_facecolor("#121826")
    ax.grid(True, alpha=0.14, color="#d9e2f2")
    for spine in ax.spines.values():
        spine.set_color("#55627a")
    ax.tick_params(colors="#d8e1ee")
    ax.title.set_color("#f5f7fb")
    ax.xaxis.label.set_color("#d8e1ee")
    ax.yaxis.label.set_color("#d8e1ee")


def plot_results(
    reference: np.ndarray,
    echoed: np.ndarray,
    cleaned: np.ndarray,
    learned_coefficients: np.ndarray,
    true_echo_path: np.ndarray,
    sample_rate: int,
    profile_name: str,
) -> None:
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(15, 9), facecolor="#0b1020")
    grid = fig.add_gridspec(2, 2, hspace=0.28, wspace=0.18)

    ax_before = fig.add_subplot(grid[0, 0])
    ax_after = fig.add_subplot(grid[0, 1])
    ax_overlay = fig.add_subplot(grid[1, 0])
    ax_taps = fig.add_subplot(grid[1, 1])

    for ax in [ax_before, ax_after, ax_overlay, ax_taps]:
        style_axes(ax)

    t = np.arange(len(reference)) / sample_rate
    ref_norm = normalize_audio(reference)
    echo_norm = normalize_audio(echoed)
    clean_norm = normalize_audio(cleaned)

    ax_before.plot(t, echo_norm, color="#ff6b6b", linewidth=1.1, label="Echoed input")
    ax_before.plot(t, ref_norm, color="#4dd0a8", linewidth=0.8, alpha=0.65, label="Reference")
    ax_before.set_title("Before: mic input vs reference")
    ax_before.set_xlabel("Time (s)")
    ax_before.set_ylabel("Amplitude")
    ax_before.legend(facecolor="#121826", edgecolor="#55627a")

    ax_after.plot(t, clean_norm, color="#77bdfb", linewidth=1.15, label="Filtered output")
    ax_after.plot(t, ref_norm, color="#f4d35e", linewidth=0.8, alpha=0.55, label="Reference")
    ax_after.set_title("After: filtered output vs reference")
    ax_after.set_xlabel("Time (s)")
    ax_after.set_ylabel("Amplitude")
    ax_after.legend(facecolor="#121826", edgecolor="#55627a")

    ax_overlay.plot(t, echo_norm, color="#ff6b6b", linewidth=1.0, alpha=0.70, label="Before")
    ax_overlay.plot(t, clean_norm, color="#77bdfb", linewidth=1.0, alpha=0.90, label="After")
    ax_overlay.set_title("Before/after comparison")
    ax_overlay.set_xlabel("Time (s)")
    ax_overlay.set_ylabel("Normalized amplitude")
    ax_overlay.legend(facecolor="#121826", edgecolor="#55627a")

    show_count = min(len(learned_coefficients), len(true_echo_path) + 180)
    taps_ms = np.arange(show_count) / sample_rate * 1000.0
    ax_taps.bar(taps_ms, learned_coefficients[:show_count], width=0.8, color="#f08c3f", alpha=0.82, label="Learned FIR taps")
    true_count = min(show_count, len(true_echo_path))
    ax_taps.plot(
        taps_ms[:true_count],
        true_echo_path[:true_count],
        color="#7cf29a",
        linewidth=2.0,
        alpha=0.9,
        label="Actual echo path",
    )
    ax_taps.set_title("Impulse response the filter learned")
    ax_taps.set_xlabel("Delay (ms)")
    ax_taps.set_ylabel("Tap amplitude")
    ax_taps.legend(facecolor="#121826", edgecolor="#55627a")

    fig.suptitle(
        f"EchoK | {profile_name}\nAdaptive FIR echo attenuation dashboard",
        color="#f5f7fb",
        fontsize=18,
        fontweight="bold",
    )
    
    plt.savefig(OUTPUT_FIGURE, dpi=180, facecolor=fig.get_facecolor(), bbox_inches="tight")


def explain_todos(filter_order: int, mu: float, sample_rate: int) -> None:
    delay_samples = int(sample_rate * ECHO_DELAY_MS / 1000)
    print("\nTODO cheat-sheet")
    print(f"1. Filter order: {filter_order} taps. It must cover the main echo delay (~{delay_samples} samples / {ECHO_DELAY_MS} ms).")
    print(f"2. Learning rate: {mu:.3f}. Higher adapts faster but can become unstable; lower is safer but slower.")
    print("3. Coefficients: each FIR tap is the echo strength at a specific delay, which is why the plot looks like an impulse response.")


def maybe_play_audio(reference: np.ndarray, echoed: np.ndarray, cleaned: np.ndarray, skip: bool) -> None:
    if skip or not HAS_PLAYBACK:
        return
    try:
        print("\nPlayback order: reference -> echoed -> cleaned")
        sd.play(reference, SAMPLE_RATE)
        sd.wait()
        sd.play(echoed, SAMPLE_RATE)
        sd.wait()
        sd.play(cleaned, SAMPLE_RATE)
        sd.wait()
    except KeyboardInterrupt:
        sd.stop()
        print("Playback interrupted.")


def save_audio(reference: np.ndarray, echoed: np.ndarray, cleaned: np.ndarray) -> None:
    if HAS_SOUNDFILE:
        sf.write(OUTPUT_REFERENCE, reference, SAMPLE_RATE)
        sf.write(OUTPUT_ECHOED, echoed, SAMPLE_RATE)
        sf.write(OUTPUT_CLEAN, cleaned, SAMPLE_RATE)
        return

    for path, audio in [
        (OUTPUT_REFERENCE, reference),
        (OUTPUT_ECHOED, echoed),
        (OUTPUT_CLEAN, cleaned),
    ]:
        pcm = np.clip(audio, -1.0, 1.0)
        pcm = (pcm * 32767).astype(np.int16)
        with wave.open(path, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(SAMPLE_RATE)
            wav_file.writeframes(pcm.tobytes())


def main() -> None:
    args = parse_args()

    print("=" * 62)
    print(" EchoK | Adaptive FIR Echo Cancellation")
    print("=" * 62)
    print(f" Profile: {args.name}")
    print(f" Style: {PROFILE_TAGLINE}")

    reference, loaded_from_file, source_label = load_reference_audio()
    if not loaded_from_file and args.name != PROFILE_NAME:
        reference = generate_personalized_source(args.name)

    echoed, true_echo_path = add_synthetic_echo(reference)

    print(f"\nSource: {'WAV file' if loaded_from_file else 'personalized synthetic source'} ({source_label})")
    print(f"Sample rate: {SAMPLE_RATE} Hz")
    print(f"Echo delay: {ECHO_DELAY_MS} ms")
    print(f"Filter order: {FILTER_ORDER} taps")
    print(f"Learning rate: {LEARNING_RATE}")

    explain_todos(FILTER_ORDER, LEARNING_RATE, SAMPLE_RATE)

    start = time.time()
    cleaned, coefficients = lms_filter(reference, echoed, FILTER_ORDER, LEARNING_RATE)
    elapsed = time.time() - start

    before_rms = rms(echoed)
    after_rms = rms(cleaned)
    improvement_db = 20 * np.log10(max(before_rms, 1e-8) / max(after_rms, 1e-8))

    print("\nRun summary")
    print(f"- Processing time: {elapsed:.2f}s")
    print(f"- Echoed RMS: {before_rms:.4f}")
    print(f"- Cleaned RMS: {after_rms:.4f}")
    print(f"- Relative attenuation: {improvement_db:.2f} dB")

    save_audio(reference, echoed, cleaned)
    maybe_play_audio(reference, echoed, cleaned, args.no_playback)
    plot_results(reference, echoed, cleaned, coefficients, true_echo_path, SAMPLE_RATE, args.name)

    print("\nOutputs")
    print(f"- Audio: {OUTPUT_REFERENCE}, {OUTPUT_ECHOED}, {OUTPUT_CLEAN}")
    print(f"- Figure: {OUTPUT_FIGURE}")

    if not args.no_show:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()
