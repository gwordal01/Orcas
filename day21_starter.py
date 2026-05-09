"""
BUILDCORED ORCAS - Day 21: UDPOscilloscope

Sender transmits simulated sensor data over UDP.
Receiver renders it as a live oscilloscope.
Packet loss can be introduced on demand.

Controls:
- l -> toggle packet loss (10% sender-side drop simulation)
- n -> toggle noise injection
- q -> quit
"""

import collections
import socket
import struct
import sys
import threading
import time

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# NETWORK CONFIG
# ============================================================

HOST = "127.0.0.1"
PORT = 5005
SEND_RATE_HZ = 200
BUFFER_SIZE = 512
PACKET_FORMAT = "!Hfd"  # uint16(seq), float(seconds), double(value)
PACKET_SIZE = struct.calcsize(PACKET_FORMAT)
LOSS_RATE = 0.10
NOISE_STD = 0.28


# ============================================================
# SHARED STATE
# ============================================================

recv_buffer = collections.deque(maxlen=BUFFER_SIZE)
arrival_intervals_ms = collections.deque(maxlen=180)

recv_lock = threading.Lock()
stats_lock = threading.Lock()

stats = {
    "sent": 0,
    "received": 0,
    "simulated_drops": 0,
    "missing_packets": 0,
    "loss_pct": 0.0,
    "last_seq": None,
    "out_of_order": 0,
    "duplicates": 0,
    "last_arrival": None,
    "jitter_ms": 0.0,
}

loss_enabled = threading.Event()
noise_enabled = threading.Event()
running = True


def refresh_loss_stats():
    expected = stats["received"] + stats["missing_packets"]
    if expected > 0:
        stats["loss_pct"] = 100.0 * stats["missing_packets"] / expected
    else:
        stats["loss_pct"] = 0.0


# ============================================================
# SENDER THREAD
# ============================================================

def sender_thread():
    """
    Simulates a sensor streaming packets across a lossy transport.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    seq = 0
    t_start = time.perf_counter()
    interval = 1.0 / SEND_RATE_HZ

    while running:
        t_now = time.perf_counter()
        elapsed = t_now - t_start

        # Primary signal: 2 Hz sine wave with a tiny harmonic so it looks lively.
        base = np.sin(2 * np.pi * 2.0 * elapsed)
        harmonic = 0.18 * np.sin(2 * np.pi * 7.0 * elapsed + 0.4)
        value = base + harmonic

        if noise_enabled.is_set():
            value += np.random.normal(0.0, NOISE_STD)

        packet = struct.pack(PACKET_FORMAT, seq % 65536, elapsed, value)
        should_drop = loss_enabled.is_set() and np.random.random() < LOSS_RATE

        if not should_drop:
            try:
                sock.sendto(packet, (HOST, PORT))
            except OSError:
                pass

        with stats_lock:
            stats["sent"] += 1
            if should_drop:
                stats["simulated_drops"] += 1

        seq += 1
        time.sleep(interval)

    sock.close()


# ============================================================
# RECEIVER
# ============================================================

def receive_packets(sock):
    """Background thread: receive UDP packets and push them into the ring buffer."""
    while running:
        try:
            sock.settimeout(0.1)
            data, _ = sock.recvfrom(1024)
            if len(data) < PACKET_SIZE:
                continue

            seq, packet_time, value = struct.unpack(PACKET_FORMAT, data[:PACKET_SIZE])
            arrival = time.perf_counter()

            with stats_lock:
                last_seq = stats["last_seq"]
                if last_seq is not None:
                    delta = (seq - last_seq) % 65536
                    if delta == 0:
                        stats["duplicates"] += 1
                    elif delta == 1:
                        pass
                    elif delta < 32768:
                        stats["missing_packets"] += delta - 1
                    else:
                        stats["out_of_order"] += 1

                if stats["last_arrival"] is not None:
                    interval_ms = (arrival - stats["last_arrival"]) * 1000.0
                    arrival_intervals_ms.append(interval_ms)
                    if len(arrival_intervals_ms) >= 6:
                        stats["jitter_ms"] = float(np.std(arrival_intervals_ms))

                stats["last_arrival"] = arrival
                stats["last_seq"] = seq
                stats["received"] += 1
                refresh_loss_stats()

            with recv_lock:
                recv_buffer.append((packet_time, value, seq))

        except socket.timeout:
            continue
        except OSError:
            continue


# ============================================================
# OSCILLOSCOPE VISUALIZATION
# ============================================================

def run_oscilloscope():
    """Render the live oscilloscope dashboard."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.bind((HOST, PORT))
    except OSError as exc:
        print(f"ERROR: Cannot bind to {HOST}:{PORT} - {exc}")
        print("Check whether another process is already using port 5005.")
        sys.exit(1)

    recv_thread = threading.Thread(target=receive_packets, args=(sock,), daemon=True)
    recv_thread.start()

    plt.style.use("dark_background")
    fig = plt.figure(figsize=(12.5, 7.2), facecolor="#06131a")
    grid = fig.add_gridspec(3, 1, height_ratios=[4.2, 1.2, 1.0], hspace=0.22)
    ax_wave = fig.add_subplot(grid[0])
    ax_loss = fig.add_subplot(grid[1])
    ax_info = fig.add_subplot(grid[2])

    fig.suptitle("UDP Oscilloscope", fontsize=18, fontweight="bold", color="#dff8ff")
    fig.text(
        0.5,
        0.925,
        "Live sine sensor stream with sequence-based loss tracking",
        ha="center",
        fontsize=10,
        color="#7dd3fc",
    )

    for axis in (ax_wave, ax_loss, ax_info):
        axis.set_facecolor("#081b24")
        for spine in axis.spines.values():
            spine.set_color("#1e3a45")

    ax_wave.set_xlim(0, BUFFER_SIZE)
    ax_wave.set_ylim(-1.8, 1.8)
    ax_wave.set_title("Waveform", loc="left", color="#c8f9ff", fontsize=12, pad=10)
    ax_wave.set_ylabel("Amplitude", color="#9ed9e3")
    ax_wave.tick_params(colors="#87c7d5")
    ax_wave.grid(True, color="#123542", alpha=0.75, linewidth=0.7)
    ax_wave.axhline(0.0, color="#295765", linewidth=1.0, linestyle="--")

    ax_loss.set_xlim(0, 119)
    ax_loss.set_ylim(0, 100)
    ax_loss.set_title("Observed Packet Loss", loc="left", color="#c8f9ff", fontsize=12, pad=8)
    ax_loss.set_ylabel("%", color="#9ed9e3")
    ax_loss.set_xlabel("Recent frames", color="#9ed9e3")
    ax_loss.tick_params(colors="#87c7d5")
    ax_loss.grid(True, color="#123542", alpha=0.65, linewidth=0.6)

    ax_info.axis("off")

    wave_line, = ax_wave.plot([], [], color="#3bff8a", linewidth=1.8)
    wave_fill = ax_wave.fill_between([], [], [], color="#22c55e", alpha=0.12)
    loss_line, = ax_loss.plot([], [], color="#f97316", linewidth=2.0)
    loss_area = ax_loss.fill_between([], [], [], color="#f97316", alpha=0.18)

    status_box = ax_wave.text(
        0.015,
        0.96,
        "",
        transform=ax_wave.transAxes,
        va="top",
        fontsize=10,
        color="#e6fbff",
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.45", facecolor="#0d2833", edgecolor="#295765", alpha=0.95),
    )
    health_box = ax_wave.text(
        0.985,
        0.96,
        "",
        transform=ax_wave.transAxes,
        va="top",
        ha="right",
        fontsize=10,
        color="#fca5a5",
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.45", facecolor="#2b1220", edgecolor="#7f1d1d", alpha=0.95),
    )
    info_text = ax_info.text(
        0.5,
        0.52,
        "",
        ha="center",
        va="center",
        fontsize=10,
        color="#b6edf7",
        fontfamily="monospace",
    )

    loss_history = collections.deque([0.0] * 120, maxlen=120)

    def update(_frame):
        nonlocal wave_fill, loss_area

        with recv_lock:
            buffer_snapshot = list(recv_buffer)

        if not buffer_snapshot:
            return wave_line, status_box, health_box, info_text, loss_line

        values = np.array([sample[1] for sample in buffer_snapshot], dtype=float)
        x = np.arange(values.size)

        wave_line.set_data(x, values)
        wave_fill.remove()
        wave_fill = ax_wave.fill_between(x, values, 0.0, color=wave_line.get_color(), alpha=0.14)

        peak = max(1.4, float(np.max(np.abs(values)) + 0.2))
        ax_wave.set_ylim(-peak, peak)

        with stats_lock:
            sent = stats["sent"]
            received = stats["received"]
            simulated_drops = stats["simulated_drops"]
            missing_packets = stats["missing_packets"]
            loss_pct = stats["loss_pct"]
            out_of_order = stats["out_of_order"]
            duplicates = stats["duplicates"]
            jitter_ms = stats["jitter_ms"]
            last_seq = stats["last_seq"]

        loss_history.append(loss_pct)
        loss_x = np.arange(len(loss_history))
        loss_y = np.array(loss_history, dtype=float)
        loss_line.set_data(loss_x, loss_y)
        loss_area.remove()
        loss_area = ax_loss.fill_between(loss_x, loss_y, 0.0, color=loss_line.get_color(), alpha=0.18)

        if loss_pct >= 15:
            wave_color = "#ff5c7a"
            health_label = "LINK HEALTH: BAD"
            health_color = "#ff8ea1"
            health_face = "#34101b"
            health_edge = "#7f1d1d"
        elif loss_pct >= 3:
            wave_color = "#ffb84d"
            health_label = "LINK HEALTH: DEGRADED"
            health_color = "#ffd38a"
            health_face = "#35240d"
            health_edge = "#9a6700"
        else:
            wave_color = "#3bff8a"
            health_label = "LINK HEALTH: CLEAN"
            health_color = "#bbf7d0"
            health_face = "#0d2818"
            health_edge = "#166534"

        wave_line.set_color(wave_color)
        loss_line.set_color("#f97316" if loss_pct >= 3 else "#38bdf8")
        health_box.set_text(f"{health_label}\nLoss {loss_pct:5.2f}%")
        health_box.set_color(health_color)
        health_box.get_bbox_patch().set_facecolor(health_face)
        health_box.get_bbox_patch().set_edgecolor(health_edge)

        modes = []
        if loss_enabled.is_set():
            modes.append(f"LOSS {int(LOSS_RATE * 100)}%")
        if noise_enabled.is_set():
            modes.append("NOISE ON")
        mode_text = " | ".join(modes) if modes else "CLEAN STREAM"

        status_box.set_text(
            f"UDP {HOST}:{PORT}\n"
            f"TX {SEND_RATE_HZ:>3} Hz | {mode_text}\n"
            f"Last seq {last_seq if last_seq is not None else '--'}"
        )

        info_text.set_text(
            f"Sent {sent:5d}   Received {received:5d}   Missing {missing_packets:4d}   "
            f"SimDrop {simulated_drops:4d}   OoO {out_of_order:3d}   Dup {duplicates:3d}   "
            f"Jitter {jitter_ms:5.2f} ms"
        )

        return wave_line, status_box, health_box, info_text, loss_line

    def on_key(event):
        global running

        if event.key == "l":
            if loss_enabled.is_set():
                loss_enabled.clear()
                print("Packet loss simulation OFF")
            else:
                loss_enabled.set()
                print(f"Packet loss simulation ON ({int(LOSS_RATE * 100)}% drop rate)")
        elif event.key == "n":
            if noise_enabled.is_set():
                noise_enabled.clear()
                print("Noise injection OFF")
            else:
                noise_enabled.set()
                print("Noise injection ON")
        elif event.key == "q":
            running = False
            plt.close(fig)

    fig.canvas.mpl_connect("key_press_event", on_key)

    ani = animation.FuncAnimation(
        fig,
        update,
        interval=50,
        blit=False,
        cache_frame_data=False,
    )

    print("\n" + "=" * 60)
    print(" UDP Oscilloscope - Day 21")
    print("=" * 60)
    print(f" Sender target : {HOST}:{PORT}")
    print(f" Packet rate   : {SEND_RATE_HZ} Hz")
    print(f" Packet format : {PACKET_FORMAT} ({PACKET_SIZE} bytes)")
    print(f" Buffer size   : {BUFFER_SIZE} samples")
    print()
    print(" Controls: click the plot, then press")
    print("   l -> toggle packet loss")
    print("   n -> toggle noise")
    print("   q -> quit")
    print()
    print(" Watch the receiver use sequence gaps to estimate packet loss.")
    print(" That is the same idea used in real sensor links and field buses.\n")

    plt.show()
    _ = ani  # Keep animation referenced for matplotlib.
    sock.close()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    send_thread = threading.Thread(target=sender_thread, daemon=True)
    send_thread.start()

    time.sleep(0.2)

    try:
        run_oscilloscope()
    except KeyboardInterrupt:
        pass
    finally:
        running = False

    print("\nUDP Oscilloscope ended.")
    print("Week 3 complete.")
