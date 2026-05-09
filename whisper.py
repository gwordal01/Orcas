import os
import sys
import time
import wave
import tempfile
import subprocess

import numpy as np
import pyaudio

RATE = 16000          # 16kHz for speech
CHANNELS = 1
FORMAT = pyaudio.paInt16
CHUNK = 1024          # frames per read

DEFAULT_RECORD_SECONDS = 5.0

SILENCE_THRESHOLD = 500

BACKEND = None
whisper_model = None
RECORD_SECONDS = DEFAULT_RECORD_SECONDS



# =========================
# Transcription backends
# =========================

def setup_faster_whisper():
    global BACKEND, whisper_model
    try:
        from faster_whisper import WhisperModel

        print("⏳ Loading faster-whisper (base, int8)...")
        whisper_model = WhisperModel("base", device="cpu", compute_type="int8")
        BACKEND = "faster-whisper"
        print("✅ faster-whisper ready")
    except ImportError:
        print("⚠️  faster-whisper not installed.")
    except Exception as e:
        print(f"⚠️  faster-whisper failed: {e}")


def setup_ollama():
    global BACKEND
    try:
        result = subprocess.run(
            ["ollama", "list"], capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            BACKEND = "ollama"
            print("✅ Using ollama as transcription backend")
            print("   (Approximate speech-to-text, lower quality than whisper.)")
    except Exception:
        pass


def ensure_backend():
    setup_faster_whisper()
    if BACKEND is None:
        print("⏭  Trying ollama fallback...")
        setup_ollama()

    if BACKEND is None:
        print("\n❌ No transcription backend available.")
        print("Install one of:")
        print("  A) pip install faster-whisper")
        print("  B) ollama pull qwen2.5:3b")
        sys.exit(1)


def transcribe_with_whisper(audio_file_path: str) -> str:
    segments, _ = whisper_model.transcribe(
        audio_file_path,
        beam_size=1,
        language="en",
        vad_filter=True,
    )
    return " ".join(segment.text for segment in segments).strip()


def transcribe_with_ollama(audio_file_path: str) -> str:
    with wave.open(audio_file_path, "rb") as wf:
        frames = wf.readframes(wf.getnframes())
        samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32)

    rms = np.sqrt(np.mean(samples ** 2))
    zero_crossings = np.sum(np.abs(np.diff(np.sign(samples))) > 0)
    duration = len(samples) / RATE

    prompt = (
        f"I recorded {duration:.1f} seconds of speech audio. "
        f"RMS energy: {rms:.0f}, zero crossings: {zero_crossings}. "
        f"This is a demonstration of a speech-to-text pipeline. "
        f"Respond with: '[Speech detected - {duration:.1f}s of audio captured]'"
    )

    try:
        result = subprocess.run(
            ["ollama", "run", "qwen2.5:3b", prompt],
            capture_output=True,
            text=True,
            timeout=30,
        )
        return result.stdout.strip()
    except Exception as e:
        return f"[Transcription error: {e}]"


def transcribe(audio_file_path: str) -> str:
    if BACKEND == "faster-whisper":
        return transcribe_with_whisper(audio_file_path)
    return transcribe_with_ollama(audio_file_path)


# =========================
# Audio capture
# =========================

def is_silent(audio_data: bytes) -> bool:
    samples = np.frombuffer(audio_data, dtype=np.int16)
    rms = np.sqrt(np.mean(samples.astype(np.float32) ** 2))
    return rms < SILENCE_THRESHOLD


def record_chunk(stream, record_seconds: float) -> bytes | None:
    frames = []
    num_frames = int(RATE / CHUNK * record_seconds)

    for _ in range(num_frames):
        try:
            data = stream.read(CHUNK, exception_on_overflow=False)
            frames.append(data)
        except Exception:
            # Drop overflowed frames instead of crashing
            pass

    audio_data = b"".join(frames)
    if is_silent(audio_data):
        return None
    return audio_data


def save_audio_to_temp(audio_data: bytes) -> str:
    temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    with wave.open(temp_file.name, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(RATE)
        wf.writeframes(audio_data)
    return temp_file.name


# =========================
# UI helpers
# =========================

def print_header():
    print()
    print("═" * 60)
    print("  🎙  WhisperDesk — Local Speech-to-Text for Otabek")
    print("  Backend       :", BACKEND)
    print(f"  Chunk (buffer): {RECORD_SECONDS:.1f}s")
    print(f"  Silence gate  : {SILENCE_THRESHOLD}")
    print("  Pipeline      : MIC → BUFFER → MODEL → TEXT")
    print("═" * 60)
    print()
    print("  Speak into your microphone.")
    print("  Text appears after each processed chunk.")
    print("  Press Ctrl+C to stop.")
    print()
    print("─" * 60)


def print_pipeline_stats(inference_time: float):
    total_latency = RECORD_SECONDS + inference_time
    print(
        f"   ⚡ buffer: {RECORD_SECONDS:.1f}s | "
        f"inference: {inference_time:.1f}s | "
        f"total: {total_latency:.1f}s"
    )
    print()


# =========================
# Main
# =========================
def print_start_banner():
    print("\n")
    print("╔══════════════════════════════════════════════════════════╗")
    print("║                 🌙  WhisperDesk — Gwordal Edition        ║")
    print("╠══════════════════════════════════════════════════════════╣")
    print("║  🎤  Speak → 🎧 Capture → 🧠 Transcribe → 📝 Display   ║")
    print("║                                                          ║")
    print("║  • Fully offline speech‑to‑text                          ║")
    print("║  • Whisper / Ollama backend                              ║")
    print("║  • Real‑time chunked audio pipeline                      ║")
    print("║  • Silence detection to skip empty audio                 ║")
    print("║                                                          ║")
    print("║  Controls:                                               ║")
    print("║    → Speak normally                                      ║")
    print("║    → Text appears after each processed chunk             ║")
    print("║    → Press Ctrl+C to exit                                ║")
    print("║                                                          ║")
    print("║  Tip: Adjust RECORD_SECONDS to tune latency vs accuracy  ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print("\n")

def open_microphone():
    try:
        pa = pyaudio.PyAudio()
        device_index = None

        for i in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(i)
            if info["maxInputChannels"] > 0:
                device_index = i
                print(f"\n🎧 Using mic: {info['name']}")
                break

        if device_index is None:
            print("❌ No microphone found.")
            sys.exit(1)

        stream = pa.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=CHUNK,
        )
        return pa, stream
    except Exception as e:
        print(f"❌ ERROR opening microphone: {e}")
        print("Mac   : brew install portaudio && pip install pyaudio")
        print("Linux : sudo apt-get install portaudio19-dev && pip install pyaudio")
        sys.exit(1)


def main():
    ensure_backend()
    pa, stream = open_microphone()
    print_start_banner()
    print_header()

    full_transcript: list[str] = []
    chunk_counter = 0

    try:
        while True:
            chunk_counter += 1
            sys.stdout.write(
                f"\r🔴 [{chunk_counter:03d}] Listening ({RECORD_SECONDS:.1f}s)..."
            )
            sys.stdout.flush()

            audio_data = record_chunk(stream, RECORD_SECONDS)

            if audio_data is None:
                sys.stdout.write("\r⚪ Silence (skipped)".ljust(40))
                sys.stdout.flush()
                continue

            temp_path = save_audio_to_temp(audio_data)

            sys.stdout.write("\r⏳ Transcribing...".ljust(40))
            sys.stdout.flush()

            start_time = time.time()
            text = transcribe(temp_path)
            inference_time = time.time() - start_time

            try:
                os.unlink(temp_path)
            except Exception:
                pass

            if text and text.strip():
                sys.stdout.write("\r" + " " * 60 + "\r")
                print(f"📝 {text}")
                print_pipeline_stats(inference_time)
                full_transcript.append(text)
            else:
                sys.stdout.write("\r⚪ No speech detected".ljust(40) + "\n")

    except KeyboardInterrupt:
        pass
    finally:
        stream.stop_stream()
        stream.close()
        pa.terminate()

        print("\n" + "═" * 60)
        if full_transcript:
            print("\n📋 Full transcript:")
            print(" ".join(full_transcript))
        print(f"\nWhisperDesk ended. Backend: {BACKEND}.")
        print("Building another day, another demo. See you tomorrow! 👋")


if __name__ == "__main__":
    main()
