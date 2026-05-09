"""
A lightweight breathing monitor with a distinct visual style and a
more responsive architecture. Audio capture runs on a background
thread, plotting stays simple, and timing uses perf_counter() for
steady elapsed measurements.

"""

from __future__ import annotations

import threading
import time
from collections import deque

import matplotlib
matplotlib.use("TkAgg")

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pyaudio
from scipy.signal import butter, lfilter, lfilter_zi


APP_TITLE = "Breath Compass"
USER_NAME = "Gwordal"

SAMPLE_RATE = 44100
CHUNK_SIZE = 1024
CHANNELS = 1
FORMAT = pyaudio.paFloat32

DISPLAY_SECONDS = 18
LOWPASS_CUTOFF_HZ = 0.5
LOWPASS_ORDER = 2
TRIGGER_LEVEL = 0.0018
RESET_RATIO = 0.72
MIN_BREATH_GAP_SECONDS = 1.5
SESSION_BPM_WINDOW_SECONDS = 30.0

ENVELOPE_RATE = SAMPLE_RATE / CHUNK_SIZE
MAX_HISTORY = int(DISPLAY_SECONDS * ENVELOPE_RATE * 1.4)

COLOR_BG = "#f4efe6"
COLOR_PANEL = "#fffaf2"
COLOR_GRID = "#d8cdbd"
COLOR_TEXT = "#3f3528"
COLOR_MUTED = "#7a6f61"
COLOR_LINE = "#0f766e"
COLOR_THRESHOLD = "#dc6b2f"
COLOR_PULSE = "#bb3e03"
COLOR_GOOD = "#2b9348"
COLOR_CALM = "#5f0f40"


def find_input_device(audio: pyaudio.PyAudio) -> int:
    try:
        return int(audio.get_default_input_device_info()["index"])
    except Exception:
        pass

    for index in range(audio.get_device_count()):
        info = audio.get_device_info_by_index(index)
        if info["maxInputChannels"] > 0:
            return int(index)

    raise RuntimeError("No microphone input device was found.")


def build_lowpass():
    nyquist = ENVELOPE_RATE / 2
    normalized_cutoff = min(LOWPASS_CUTOFF_HZ / nyquist, 0.95)
    return butter(LOWPASS_ORDER, normalized_cutoff, btype="low")


class BreathCompass:
    def __init__(self):
        self.audio = pyaudio.PyAudio()
        self.device_index = find_input_device(self.audio)
        self.stream = self.audio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            input=True,
            input_device_index=self.device_index,
            frames_per_buffer=CHUNK_SIZE,
        )

        self.filter_b, self.filter_a = build_lowpass()
        self.filter_state = lfilter_zi(self.filter_b, self.filter_a) * 0.0

        self.lock = threading.Lock()
        self.running = True
        self.started_at = time.perf_counter()
        self.last_breath_at = 0.0
        self.above_trigger = False
        self.current_bpm = 0.0
        self.total_breaths = 0
        self.last_level = 0.0
        self.status = "warming up"

        self.samples: deque[tuple[float, float]] = deque(maxlen=MAX_HISTORY)
        self.breath_marks: deque[float] = deque(maxlen=64)

    def audio_loop(self):
        while self.running:
            try:
                payload = self.stream.read(CHUNK_SIZE, exception_on_overflow=False)
                chunk = np.frombuffer(payload, dtype=np.float32)
            except Exception:
                continue

            rms = float(np.sqrt(np.mean(chunk ** 2)))
            filtered, self.filter_state = lfilter(
                self.filter_b,
                self.filter_a,
                [rms],
                zi=self.filter_state,
            )
            level = float(abs(filtered[0]))
            timestamp = time.perf_counter() - self.started_at

            with self.lock:
                self.samples.append((timestamp, level))
                self.last_level = level
                self._detect_breath(timestamp, level)

    def _detect_breath(self, timestamp: float, level: float):
        ready = (timestamp - self.last_breath_at) >= MIN_BREATH_GAP_SECONDS

        if not self.above_trigger and level >= TRIGGER_LEVEL and ready:
            self.above_trigger = True
            self.last_breath_at = timestamp
            self.total_breaths += 1
            self.breath_marks.append(timestamp)
            self.current_bpm = self._compute_bpm(timestamp)
            self.status = self._pace_label(self.current_bpm)
        elif self.above_trigger and level < TRIGGER_LEVEL * RESET_RATIO:
            self.above_trigger = False
            self.status = "listening"
        elif self.current_bpm > 0:
            self.status = self._pace_label(self.current_bpm)
        else:
            self.status = "listening"

    def _compute_bpm(self, now_ts: float) -> float:
        recent = [mark for mark in self.breath_marks if now_ts - mark <= SESSION_BPM_WINDOW_SECONDS]
        if len(recent) < 2:
            return 0.0

        intervals = [recent[i] - recent[i - 1] for i in range(1, len(recent))]
        mean_gap = sum(intervals) / len(intervals)
        return 60.0 / mean_gap if mean_gap > 0 else 0.0

    @staticmethod
    def _pace_label(bpm: float) -> str:
        if bpm <= 0:
            return "listening"
        if bpm < 8:
            return "very slow"
        if bpm <= 14:
            return "steady"
        if bpm <= 20:
            return "energized"
        return "too fast"

    def snapshot(self):
        with self.lock:
            return {
                "samples": list(self.samples),
                "breath_marks": list(self.breath_marks),
                "current_bpm": self.current_bpm,
                "total_breaths": self.total_breaths,
                "status": self.status,
                "elapsed": time.perf_counter() - self.started_at,
            }

    def stop(self):
        self.running = False
        try:
            self.stream.stop_stream()
            self.stream.close()
        finally:
            self.audio.terminate()


def build_figure():
    fig = plt.figure(figsize=(11.5, 7), facecolor=COLOR_BG)
    fig.canvas.manager.set_window_title(APP_TITLE)

    ax_wave = fig.add_axes([0.08, 0.38, 0.84, 0.46], facecolor=COLOR_PANEL)
    ax_bpm = fig.add_axes([0.08, 0.12, 0.24, 0.16], facecolor=COLOR_PANEL)
    ax_count = fig.add_axes([0.38, 0.12, 0.24, 0.16], facecolor=COLOR_PANEL)
    ax_clock = fig.add_axes([0.68, 0.12, 0.24, 0.16], facecolor=COLOR_PANEL)

    for axis in (ax_bpm, ax_count, ax_clock):
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_color(COLOR_GRID)

    ax_wave.set_xlim(-DISPLAY_SECONDS, 0)
    ax_wave.set_ylim(0, 0.02)
    ax_wave.set_title(f"{USER_NAME}'s breathing trace", loc="left", color=COLOR_TEXT, fontsize=13, pad=10)
    ax_wave.set_xlabel("seconds from now", color=COLOR_MUTED)
    ax_wave.set_ylabel("breath envelope", color=COLOR_MUTED)
    ax_wave.grid(True, linestyle="--", linewidth=0.6, color=COLOR_GRID, alpha=0.9)
    ax_wave.tick_params(colors=COLOR_MUTED)
    for spine in ax_wave.spines.values():
        spine.set_color(COLOR_GRID)

    wave_line, = ax_wave.plot([], [], color=COLOR_LINE, linewidth=2.4)
    threshold_line = ax_wave.axhline(TRIGGER_LEVEL, color=COLOR_THRESHOLD, linewidth=1.2, linestyle=":")
    breath_scatter = ax_wave.scatter([], [], color=COLOR_PULSE, s=28, zorder=4)

    fig.text(0.5, 0.92, APP_TITLE, ha="center", color=COLOR_TEXT, fontsize=18, fontweight="bold")
    fig.text(
        0.5,
        0.89,
        "Here is what I'm picking up from the mic in real time. Try breathing smoothly near the mic to see the wave respond!",
        ha="center",
        color=COLOR_MUTED,
        fontsize=9,
    )

    bpm_value = ax_bpm.text(0.5, 0.6, "--", ha="center", va="center", fontsize=30, color=COLOR_GOOD, fontweight="bold")
    ax_bpm.text(0.5, 0.18, "breaths per min", ha="center", va="center", fontsize=10, color=COLOR_MUTED)

    count_value = ax_count.text(0.5, 0.6, "0", ha="center", va="center", fontsize=30, color=COLOR_LINE, fontweight="bold")
    ax_count.text(0.5, 0.18, "detected breaths", ha="center", va="center", fontsize=10, color=COLOR_MUTED)

    clock_value = ax_clock.text(0.5, 0.6, "0:00", ha="center", va="center", fontsize=30, color=COLOR_CALM, fontweight="bold")
    ax_clock.text(0.5, 0.18, "session clock", ha="center", va="center", fontsize=10, color=COLOR_MUTED)

    status_text = ax_wave.text(
        0.015,
        0.92,
        "Status: listening",
        transform=ax_wave.transAxes,
        color=COLOR_TEXT,
        fontsize=11,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#f9f4ec", "edgecolor": COLOR_GRID},
    )

    return {
        "fig": fig,
        "ax_wave": ax_wave,
        "wave_line": wave_line,
        "threshold_line": threshold_line,
        "breath_scatter": breath_scatter,
        "bpm_value": bpm_value,
        "count_value": count_value,
        "clock_value": clock_value,
        "status_text": status_text,
    }


def format_elapsed(elapsed: float) -> str:
    whole_seconds = int(elapsed)
    minutes, seconds = divmod(whole_seconds, 60)
    return f"{minutes}:{seconds:02d}"


def main():
    print(f"\n{APP_TITLE}")
    print("-" * len(APP_TITLE))
    print("Welcome here! This is a simple breathing monitor that detects breaths based on microphone input and visualizes them in real time.")
    print(f"Trigger level: {TRIGGER_LEVEL}")
    print(f"Low-pass cutoff: {LOWPASS_CUTOFF_HZ} Hz")
    print("Tip: breathe near the mic with smooth inhales and exhales.\n")

    compass = BreathCompass()
    handles = build_figure()

    worker = threading.Thread(target=compass.audio_loop, daemon=True)
    worker.start()

    def update(_frame):
        snap = compass.snapshot()
        samples = snap["samples"]
        elapsed = snap["elapsed"]

        if samples:
            visible = [(ts, level) for ts, level in samples if elapsed - ts <= DISPLAY_SECONDS]
            if visible:
                x_values = np.array([ts - elapsed for ts, _ in visible], dtype=float)
                y_values = np.array([level for _, level in visible], dtype=float)
                handles["wave_line"].set_data(x_values, y_values)

                y_limit = max(TRIGGER_LEVEL * 1.8, float(y_values.max()) * 1.3, 0.008)
                handles["ax_wave"].set_ylim(0, y_limit)

                breath_marks = [mark - elapsed for mark in snap["breath_marks"] if elapsed - mark <= DISPLAY_SECONDS]
                if breath_marks:
                    handles["breath_scatter"].set_offsets(
                        np.column_stack([breath_marks, np.full(len(breath_marks), TRIGGER_LEVEL)])
                    )
                else:
                    handles["breath_scatter"].set_offsets(np.empty((0, 2)))

        handles["threshold_line"].set_ydata([TRIGGER_LEVEL, TRIGGER_LEVEL])
        handles["bpm_value"].set_text(f"{snap['current_bpm']:.1f}" if snap["current_bpm"] > 0 else "--")
        handles["count_value"].set_text(str(snap["total_breaths"]))
        handles["clock_value"].set_text(format_elapsed(elapsed))
        handles["status_text"].set_text(f"Status: {snap['status']}")

        return (
            handles["wave_line"],
            handles["threshold_line"],
            handles["breath_scatter"],
            handles["bpm_value"],
            handles["count_value"],
            handles["clock_value"],
            handles["status_text"],
        )

    animator = animation.FuncAnimation(
        handles["fig"],
        update,
        interval=50,
        blit=False,
        cache_frame_data=False,
    )

    try:
        plt.show()
    except KeyboardInterrupt:
        pass
    finally:
        _ = animator
        compass.stop()
        worker.join(timeout=1.0)
        final = compass.snapshot()
        print(f"\nSession length: {format_elapsed(final['elapsed'])}")
        print(f"Breaths detected: {final['total_breaths']}")
        if final["current_bpm"] > 0:
            print(f"Last BPM estimate: {final['current_bpm']:.1f}")


if __name__ == "__main__":
    main()
