"""
Gwordal — Day 15: AudioScope
===================================
Dark‑mode FFT spectrum analyzer — visualize your microphone input.

Features:
- Real-time mic capture
- FFT spectrum updates live
- 6 tuned frequency bands (sub-bass, bass, low-mid, mid, high-mid, treble/air)
- Peak frequency detection with marker line
- Smooth dark UI with neon accents
"""

import pyaudio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sys

# ============================================================
# AUDIO SETUP
# ============================================================

RATE = 44100
CHUNK = 2048
FORMAT = pyaudio.paFloat32
CHANNELS = 1

try:
    pa = pyaudio.PyAudio()
    device_index = None
    for i in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(i)
        if info['maxInputChannels'] > 0:
            device_index = i
            print(f"Using mic: {info['name']}")
            break
    if device_index is None:
        print("ERROR: No microphone found.")
        sys.exit(1)

    stream = pa.open(
        format=FORMAT, channels=CHANNELS, rate=RATE,
        input=True, input_device_index=device_index,
        frames_per_buffer=CHUNK,
    )
except Exception as e:
    print(f"ERROR opening mic: {e}")
    sys.exit(1)


# ============================================================
# FFT SETUP
# ============================================================

freqs = np.fft.rfftfreq(CHUNK, d=1.0/RATE)

# Expanded bands (6 total)
BANDS = [
    ("Sub-bass", 20, 80, "#00ff99"),
    ("Bass", 80, 250, "#00ccff"),
    ("Low-mid", 250, 500, "#33ff33"),
    ("Mid", 500, 2000, "#ffff33"),
    ("High-mid", 2000, 6000, "#ff9933"),
    ("Treble/Air", 6000, 20000, "#ff3399"),
]

def band_level(fft_magnitudes, freq_array, band):
    name, low, high, color = band
    mask = (freq_array >= low) & (freq_array < high)
    if mask.sum() == 0:
        return 0.0
    return float(np.mean(fft_magnitudes[mask]))

def find_peak_frequency(fft_magnitudes, freq_array, min_freq=50):
    valid = freq_array >= min_freq
    if not valid.any():
        return 0.0
    valid_mags = fft_magnitudes.copy()
    valid_mags[~valid] = 0
    peak_idx = np.argmax(valid_mags)
    return float(freq_array[peak_idx])


# ============================================================
# MATPLOTLIB DARK MODE UI
# ============================================================

plt.style.use("dark_background")
fig, (ax_spec, ax_bands) = plt.subplots(2, 1, figsize=(12, 8))
fig.patch.set_facecolor("#0a0a0a")
fig.suptitle("AudioScope — Otabek Edition (Dark Mode)", fontsize=14, fontweight="bold", color="#00ffff")

# Spectrum plot
ax_spec.set_xscale("log")
ax_spec.set_xlim(20, 20000)
ax_spec.set_ylim(0, 0.05)
ax_spec.set_xlabel("Frequency (Hz)", color="#aaaaaa")
ax_spec.set_ylabel("Magnitude", color="#aaaaaa")
ax_spec.set_title("Live Frequency Spectrum", color="#00ffff")
ax_spec.grid(True, which="both", alpha=0.25, color="#444")

spectrum_line, = ax_spec.plot([], [], color='#00ffcc', linewidth=1.5, alpha=0.9)
peak_marker = ax_spec.axvline(0, color="#ff0066", linestyle="--", linewidth=1.2, alpha=0.8)

# Band level bar chart
band_names = [b[0] for b in BANDS]
band_colors = [b[3] for b in BANDS]
bars = ax_bands.bar(band_names, [0] * len(BANDS), color=band_colors)
ax_bands.set_ylim(0, 0.05)
ax_bands.set_ylabel("Average Magnitude", color="#aaaaaa")
ax_bands.set_title("Frequency Bands", color="#00ffff")
ax_bands.grid(True, axis='y', alpha=0.25, color="#444")

# Peak frequency text
peak_text = ax_spec.text(
    0.02, 0.92, "Peak: --",
    transform=ax_spec.transAxes, fontsize=11, fontweight='bold',
    color='#00ffff',
    bbox=dict(boxstyle='round', facecolor='#111', alpha=0.8)
)

plt.tight_layout()


# ============================================================
# ANIMATION UPDATE
# ============================================================

smooth_db = None
SMOOTH_ALPHA = 0.3  # smoothing factor

def update(frame_num):
    global smooth_db
    try:
        audio_data = stream.read(CHUNK, exception_on_overflow=False)
        samples = np.frombuffer(audio_data, dtype=np.float32)

        # Blackman window for smoother spectrum
        windowed = samples * np.blackman(len(samples))

        fft_result = np.fft.rfft(windowed)
        magnitudes = np.abs(fft_result) / CHUNK

        # Exponential smoothing
        if smooth_db is None:
            smooth_db = magnitudes
        else:
            smooth_db = SMOOTH_ALPHA * magnitudes + (1 - SMOOTH_ALPHA) * smooth_db

        # Update spectrum line
        spectrum_line.set_data(freqs, smooth_db)

        peak_mag = max(smooth_db.max(), 0.001)
        ax_spec.set_ylim(0, peak_mag * 1.2)

        # Band levels
        levels = [band_level(smooth_db, freqs, band) for band in BANDS]
        for bar, level in zip(bars, levels):
            bar.set_height(level)

        max_band = max(max(levels), 0.001)
        ax_bands.set_ylim(0, max_band * 1.2)

        # Peak detection
        peak_freq = find_peak_frequency(smooth_db, freqs)
        if peak_freq > 0:
            peak_text.set_text(f"Peak: {peak_freq:.0f} Hz")
            peak_marker.set_xdata([peak_freq, peak_freq])
        else:
            peak_text.set_text("Peak: --")

    except Exception as e:
        peak_text.set_text(f"Error: {e}")

    return spectrum_line, peak_text, peak_marker, *bars


# ============================================================
# RUN
# ============================================================

print("\nAudioScope — Dark Mode Edition is running!")
print(f"Sample rate: {RATE} Hz | FFT size: {CHUNK} | Resolution: {RATE/CHUNK:.1f} Hz/bin")
print(f"Bands: {', '.join(b[0] for b in BANDS)}")
print("Try: whistle, clap, speak, play music")
print("Close the plot window to quit.\n")

try:
    ani = animation.FuncAnimation(
        fig, update,
        interval=int(1000 * CHUNK / RATE),
        blit=False, cache_frame_data=False
    )
    plt.show()
except KeyboardInterrupt:
    pass
finally:
    stream.stop_stream()
    stream.close()
    pa.terminate()
    print("\nAudioScope ended. Nice hearing you!")
