import argparse
import struct

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np


SAMPLES_PER_BIT = 24
WINDOW = 220
FRAME_STEP = 4
DEFAULT_STRETCH = 18


def byte_to_bits(value):
    return [(value >> shift) & 1 for shift in range(7, -1, -1)]


def bits_to_int(bits):
    value = 0
    for bit in bits:
        value = (value << 1) | bit
    return value


def start_segment():
    return ("START", [1, 1, 0, 0], [1, 1, 1, 0], "START")


def stop_segment():
    return ("STOP", [0, 0, 1, 1], [0, 1, 1, 1], "STOP")


def build_clock_waveform(stretch_low=0):
    low_pre = SAMPLES_PER_BIT // 4 + stretch_low
    high = SAMPLES_PER_BIT // 2
    low_post = SAMPLES_PER_BIT - (SAMPLES_PER_BIT // 4) - high
    return [0] * low_pre + [1] * high + [0] * low_post


def bits_to_waveform(bits, stretch_bit_index=None, stretch_low=0):
    sda = []
    scl = []
    for index, bit in enumerate(bits):
        extra = stretch_low if stretch_bit_index == index else 0
        bit_clock = build_clock_waveform(stretch_low=extra)
        sda.extend([bit] * len(bit_clock))
        scl.extend(bit_clock)
    return sda, scl


def ack_segment(ack=True, stretch_low=0, source="slave"):
    level = 0 if ack else 1
    scl = build_clock_waveform(stretch_low=stretch_low)
    sda = [level] * len(scl)
    label = "ACK" if ack else "NACK"
    if stretch_low:
        note = f"{label} + STRETCH"
    elif source == "master" and not ack:
        note = "MASTER NACK"
    else:
        note = label
    return (label, sda, scl, note)


def encode_i2c_transaction(
    device_addr,
    register_addr,
    data_bytes,
    *,
    read=False,
    nack_after=None,
    stretch_ack_indices=None,
):
    nack_after = set() if nack_after is None else set(nack_after)
    stretch_ack_indices = set() if stretch_ack_indices is None else set(stretch_ack_indices)

    segments = [start_segment()]
    addr_rw = ((device_addr & 0x7F) << 1) | (1 if read else 0)
    payload = [addr_rw, register_addr, *data_bytes]
    labels = ["ADDR+RW", "REG"] + [f"DATA[{index}]" for index in range(len(data_bytes))]
    notes = [
        f"ADDR=0x{device_addr:02X} {'R' if read else 'W'}",
        f"REG=0x{register_addr:02X}",
        *[f"DATA=0x{value:02X}" for value in data_bytes],
    ]

    aborted = False
    for ack_index, (value, label, note) in enumerate(zip(payload, labels, notes)):
        sda, scl = bits_to_waveform(byte_to_bits(value))
        segments.append((label, sda, scl, note))

        if read and ack_index == len(payload) - 1:
            ack = False
            source = "master"
        else:
            ack = ack_index not in nack_after
            source = "slave"

        stretch_low = DEFAULT_STRETCH if ack_index in stretch_ack_indices else 0
        segments.append(ack_segment(ack=ack, stretch_low=stretch_low, source=source))

        if not ack and source == "slave":
            aborted = True
            break

    if aborted:
        segments.append(("ABORT", [1, 1, 1, 1], [0, 0, 0, 0], "MASTER ABORTS"))

    segments.append(stop_segment())
    return segments


def build_full_waveform(segments):
    sda_all = []
    scl_all = []
    labels = []
    cursor = 0
    for _, sda, scl, note in segments:
        labels.append((cursor, note))
        sda_all.extend(sda)
        scl_all.extend(scl)
        cursor += len(sda)
    return np.asarray(sda_all), np.asarray(scl_all), labels


def sample_bits(sda_bits, scl_bits):
    bits = []
    scl_array = np.asarray(scl_bits)
    rising = np.where((scl_array[1:] == 1) & (scl_array[:-1] == 0))[0] + 1
    for edge in rising:
        high_end = edge
        while high_end < len(scl_bits) and scl_bits[high_end] == 1:
            high_end += 1
        sample_index = edge + (high_end - edge) // 2
        bits.append(int(sda_bits[sample_index]))
    return bits


def decode_i2c_segments(segments):
    decoded_bytes = []
    ack_states = []
    details = []

    for label, sda, scl, note in segments:
        if label in {"START", "STOP", "ABORT"}:
            details.append(f"[{note}]")
            continue
        if label in {"ACK", "NACK"}:
            ack_states.append(label)
            details.append(f"  -> {note}")
            continue

        bits = sample_bits(sda, scl)
        value = bits_to_int(bits[:8]) if len(bits) >= 8 else None
        decoded_bytes.append((label, value))
        details.append(f"  {label}: 0x{value:02X} ({value:08b})")

    result = {"details": details, "bytes": decoded_bytes, "acks": ack_states, "data_bytes": []}
    if decoded_bytes:
        addr_rw = decoded_bytes[0][1]
        result["device_addr"] = addr_rw >> 1
        result["read"] = bool(addr_rw & 1)
    if len(decoded_bytes) >= 2:
        result["register_addr"] = decoded_bytes[1][1]
    if len(decoded_bytes) >= 3:
        result["data_bytes"] = [value for _, value in decoded_bytes[2:]]
    return result


def print_transaction(name, segments):
    print("\n" + "=" * 72)
    print(name)
    print("=" * 72)
    for label, sda, scl, note in segments:
        sda_preview = "".join(str(bit) for bit in sda[:16])
        scl_preview = "".join(str(bit) for bit in scl[:16])
        print(f"{label:<8} | SCL {scl_preview:<16} | SDA {sda_preview:<16} | {note}")

    decoded = decode_i2c_segments(segments)
    print("\nDecoded:")
    for line in decoded["details"]:
        print(line)

    if "device_addr" in decoded:
        print(f"\nRecovered device: 0x{decoded['device_addr']:02X}")
        if "register_addr" in decoded:
            print(f"Recovered register: 0x{decoded['register_addr']:02X}")
        print(f"Recovered data: {[f'0x{value:02X}' for value in decoded['data_bytes']]}")
    return decoded


def save_waveform(segments, title, output_path):
    sda_full, scl_full, labels = build_full_waveform(segments)
    x = np.arange(len(sda_full))

    fig, (ax_scl, ax_sda) = plt.subplots(2, 1, figsize=(14, 5), sharex=True)
    fig.suptitle(title, fontsize=13, fontweight="bold")

    ax_scl.step(x, scl_full, where="post", linewidth=1.8, color="#f59e0b")
    ax_sda.step(x, sda_full, where="post", linewidth=1.8, color="#22c55e")

    for ax, label in ((ax_scl, "SCL"), (ax_sda, "SDA")):
        ax.set_ylim(-0.25, 1.35)
        ax.set_ylabel(label)
        ax.set_yticks([0, 1], labels=["LOW", "HIGH"])
        ax.grid(True, alpha=0.25)

    for position, note in labels:
        ax_scl.axvline(position, color="#94a3b8", linewidth=0.5, alpha=0.25)
        ax_scl.text(position + 2, 1.16, note, fontsize=7, rotation=90, va="bottom")

    ax_sda.set_xlabel("Sample index")
    plt.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def animate_transaction(segments, title):
    sda_full, scl_full, labels = build_full_waveform(segments)
    total = len(sda_full)

    fig, (ax_scl, ax_sda) = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.patch.set_facecolor("#0a0a0f")

    for ax, name in ((ax_scl, "SCL"), (ax_sda, "SDA")):
        ax.set_facecolor("#0d0d14")
        ax.set_ylim(-0.3, 1.5)
        ax.set_ylabel(name, fontweight="bold", color="white")
        ax.set_yticks([0, 1], labels=["LOW", "HIGH"])
        ax.grid(True, alpha=0.2)
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333")

    ax_sda.set_xlabel("Sample index", color="white")
    scl_line, = ax_scl.step([], [], where="post", linewidth=2.2, color="#f59e0b")
    sda_line, = ax_sda.step([], [], where="post", linewidth=2.2, color="#22c55e")
    note_text = ax_scl.text(
        0.02,
        1.15,
        "",
        transform=ax_scl.transAxes,
        fontsize=10,
        color="white",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="#1e293b", edgecolor="#475569", alpha=0.85),
    )

    def update(frame):
        start = min(frame * FRAME_STEP, max(0, total - WINDOW))
        end = min(total, start + WINDOW)
        x = np.arange(start, start + WINDOW)
        scl_slice = scl_full[start:end]
        sda_slice = sda_full[start:end]
        if len(scl_slice) < WINDOW:
            scl_slice = np.pad(scl_slice, (0, WINDOW - len(scl_slice)), constant_values=np.nan)
            sda_slice = np.pad(sda_slice, (0, WINDOW - len(sda_slice)), constant_values=np.nan)

        scl_line.set_data(x, scl_slice)
        sda_line.set_data(x, sda_slice)
        ax_scl.set_xlim(start, start + WINDOW)
        ax_sda.set_xlim(start, start + WINDOW)

        current_note = ""
        for position, note in labels:
            if position <= start + WINDOW // 2:
                current_note = note
        note_text.set_text(current_note)
        return scl_line, sda_line, note_text

    frames = max(1, (max(total - WINDOW, 0) // FRAME_STEP) + 20)
    ani = animation.FuncAnimation(fig, update, frames=frames, interval=60, repeat=True)
    plt.tight_layout()
    plt.show()
    return ani


def make_normal_demo():
    return encode_i2c_transaction(0x48, 0x1A, [0x42], read=False)


def make_nack_demo():
    return encode_i2c_transaction(0x48, 0x1A, [0x42], nack_after={0})


def make_stretch_demo():
    return encode_i2c_transaction(0x68, 0x3B, [0x03, 0xE8], stretch_ack_indices={1})


def make_mpu_demo():
    packed = struct.pack(">hhhhhhh", 1000, -250, 16384, 12, -34, 56, -78)
    return encode_i2c_transaction(0x68, 0x3B, list(packed), read=False)


def run_scenario(name, *, show_plot=True, save_path=None):
    if name == "normal":
        title = "Normal I2C Write"
        segments = make_normal_demo()
    elif name == "nack":
        title = "Address NACK Simulation"
        segments = make_nack_demo()
    elif name == "stretch":
        title = "Clock Stretching Simulation"
        segments = make_stretch_demo()
    elif name == "mpu":
        title = "MPU6050 Style Burst Write"
        segments = make_mpu_demo()
    else:
        raise ValueError(f"Unknown scenario: {name}")

    decoded = print_transaction(title, segments)

    if name == "mpu" and decoded["data_bytes"]:
        unpacked = struct.unpack(">hhhhhhh", bytes(decoded["data_bytes"]))
        print(f"Unpacked payload: {unpacked}")

    if save_path:
        save_waveform(segments, title, save_path)
        print(f"Saved waveform: {save_path}")

    if show_plot:
        animate_transaction(segments, title)


def build_parser():
    parser = argparse.ArgumentParser(description="I2C playground with ACK/NACK and clock stretching.")
    parser.add_argument(
        "--scenario",
        choices=["normal", "nack", "stretch", "mpu", "all"],
        default="all",
        help="Which demo scenario to run.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Skip the live matplotlib animation window.",
    )
    parser.add_argument(
        "--save",
        metavar="PNG_PATH",
        help="Save the selected scenario waveform as a PNG.",
    )
    return parser


def main():
    args = build_parser().parse_args()

    if args.scenario == "all":
        for name in ("normal", "nack", "stretch", "mpu"):
            run_scenario(name, show_plot=False, save_path=None)
        if not args.no_show:
            animate_transaction(make_stretch_demo(), "Clock Stretching Simulation")
    else:
        run_scenario(args.scenario, show_plot=not args.no_show, save_path=args.save)


if __name__ == "__main__":
    main()
