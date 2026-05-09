"""
BUILDCORED ORCAS — Day 19: I2CPlayground (COMPLETED)
Complete I2C protocol simulation with animated waveforms.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

SCL_FREQ = 100_000    
SAMPLES_PER_BIT = 20  


def bits_to_waveform(bits):
    """Convert bits to SDA/SCL arrays. SCL pulses for each bit."""
    sda = []
    scl = []
    for bit in bits:
        sda += [bit] * SAMPLES_PER_BIT
        scl += [0] * (SAMPLES_PER_BIT // 4) + \
               [1] * (SAMPLES_PER_BIT // 2) + \
               [0] * (SAMPLES_PER_BIT // 4)
    return sda, scl

def generate_ack(ack=True):
    """ACK: SDA LOW during SCL HIGH. NACK: SDA HIGH during SCL HIGH."""
    sda_val = 0 if ack else 1
    sda = [sda_val] * SAMPLES_PER_BIT
    scl = [0] * (SAMPLES_PER_BIT // 4) + \
          [1] * (SAMPLES_PER_BIT // 2) + \
          [0] * (SAMPLES_PER_BIT // 4)
    return sda, scl

def generate_stretched_ack():
    """TODO #2: Simulate Slave Clock Stretching by extending the SCL LOW phase."""
    sda = [0] * (SAMPLES_PER_BIT * 2) 
    scl = [0] * (SAMPLES_PER_BIT) + \
          [0] * (SAMPLES_PER_BIT // 2) + \
          [1] * (SAMPLES_PER_BIT // 2) + \
          [0] * (SAMPLES_PER_BIT // 2)
    return sda, scl

def encode_i2c_transaction(device_addr, register_addr, data_bytes, read=False):
    """Encodes a complete transaction into segments."""
    segments = []

    segments.append(("START", [1, 1, 0, 0], [1, 1, 1, 0], "START"))

    addr_bits = [(device_addr >> (6 - i)) & 1 for i in range(7)]
    rw_bit = 1 if read else 0
    addr_byte = addr_bits + [rw_bit]
    sda_seq, scl_seq = bits_to_waveform(addr_byte)
    segments.append(("ADDR+RW", sda_seq, scl_seq, f"ADDR=0x{device_addr:02X} {'R' if read else 'W'}"))

    ack_sda, ack_scl = generate_ack(ack=True)
    segments.append(("ACK", ack_sda, ack_scl, "ACK"))

    reg_bits = [(register_addr >> (7 - i)) & 1 for i in range(8)]
    sda_seq, scl_seq = bits_to_waveform(reg_bits)
    segments.append(("REG", sda_seq, scl_seq, f"REG=0x{register_addr:02X}"))

    ack_sda, ack_scl = generate_ack(ack=True)
    segments.append(("ACK", ack_sda, ack_scl, "ACK"))

    for i, byte in enumerate(data_bytes):
        data_bits = [(byte >> (7 - j)) & 1 for j in range(8)]
        sda_seq, scl_seq = bits_to_waveform(data_bits)
        segments.append((f"DATA[{i}]", sda_seq, scl_seq, f"DATA=0x{byte:02X}"))

        is_last = (i == len(data_bytes) - 1)
        ack = not (read and is_last)
        ack_sda, ack_scl = generate_ack(ack=ack)
        label = "ACK" if ack else "NACK"
        segments.append((label, ack_sda, ack_scl, label))

    segments.append(("STOP", [0, 0, 1, 1], [0, 1, 1, 1], "STOP"))
    return segments

def simulate_nack_transaction():
    """Simulates a device not responding (NACK after address)."""
    device_addr = 0x12 
    segments = []
    segments.append(("START", [1, 1, 0, 0], [1, 1, 1, 0], "START"))
    
    addr_bits = [(device_addr >> (6 - i)) & 1 for i in range(7)] + [0]
    sda_seq, scl_seq = bits_to_waveform(addr_bits)
    segments.append(("ADDR+W", sda_seq, scl_seq, f"ADDR=0x{device_addr:02X} W"))
    
    nack_sda, nack_scl = generate_ack(ack=False)
    segments.append(("NACK", nack_sda, nack_scl, "NACK: DEVICE NOT FOUND"))
    
    segments.append(("STOP", [0, 0, 1, 1], [0, 1, 1, 1], "STOP"))
    return segments


def decode_i2c_segments(segments):
    decoded = []
    for label, sda_bits, scl_bits, annotation in segments:
        if label in ("START", "STOP"):
            decoded.append(f"[{label}]")
        elif label in ("ACK", "NACK", "STRETCH"):
            decoded.append(f"  → {label}")
        else:
            bits = []
            i = 0
            while i < len(sda_bits):
                if i < len(scl_bits) and scl_bits[i] == 1:
                    bits.append(sda_bits[i])
                    while i < len(scl_bits) and scl_bits[i] == 1: i += 1
                else: i += 1
            if len(bits) >= 8:
                val = 0
                for b in bits[:8]: val = (val << 1) | b
                decoded.append(f"  {label}: 0x{val:02X}")
    return decoded

def build_full_waveform(segments):
    all_sda, all_scl, boundaries, labels = [], [], [0], []
    for label, sda, scl, ann in segments:
        all_sda.extend(sda)
        all_scl.extend(scl)
        boundaries.append(len(all_sda))
        labels.append((boundaries[-2], ann))
    return np.array(all_sda), np.array(all_scl), labels

def animate_transaction(segments, title="I2C"):
    sda_full, scl_full, labels = build_full_waveform(segments)
    N, WINDOW = len(sda_full), 200
    fig, (ax_scl, ax_sda) = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
    fig.suptitle(title, color='white', fontweight='bold')
    fig.patch.set_facecolor('#0a0a0f')

    for ax, name, color in zip([ax_scl, ax_sda], ["SCL", "SDA"], ['#f59e0b', '#22c55e']):
        ax.set_ylim(-0.3, 1.5)
        ax.set_ylabel(name, color='white')
        ax.set_facecolor("#0d0d14")
        ax.tick_params(colors='white')
        ax.grid(True, alpha=0.1)

    scl_line, = ax_scl.step([], [], color='#f59e0b', lw=2, where='post')
    sda_line, = ax_sda.step([], [], color='#22c55e', lw=2, where='post')
    ann_text = ax_scl.text(0.02, 1.1, "", transform=ax_scl.transAxes, color='white', 
                           bbox=dict(facecolor='#333', alpha=0.8))

    def update(frame):
        start = min(frame * 5, N - WINDOW)
        end = start + WINDOW
        
        scl_slice = scl_full[start:end]
        sda_slice = sda_full[start:end]
        
        x = np.arange(start, start + len(scl_slice))

        scl_line.set_data(x, scl_slice)
        sda_line.set_data(x, sda_slice)
        
        ax_scl.set_xlim(start, start + WINDOW)
        ax_sda.set_xlim(start, start + WINDOW)

        for pos, lbl in labels:
            if pos <= start + WINDOW // 2:
                ann_text.set_text(lbl)

        return scl_line, sda_line, ann_text

    ani = animation.FuncAnimation(fig, update, frames=N//5, interval=50, repeat=True)
    plt.show()


def main():
    print("🔌 I2CPlayground Initialized...")

    mpu_data = [0x3B, 0x01, 0x42, 0x00, 0x12, 0x34, 0xFF, 0xEE, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06]
    segments = encode_i2c_transaction(0x68, 0x3B, mpu_data)
    
    s_sda, s_scl = generate_stretched_ack()
    segments[2] = ("STRETCH", s_sda, s_scl, "CLOCK STRETCHING")

    print("\nDecoded MPU6050 Burst Read:")
    for line in decode_i2c_segments(segments): print(line)

    animate_transaction(segments, "MPU6050 14-Byte Read with Clock Stretching")

    print("\nRunning NACK Simulation (Device not found)...")
    nack_seg = simulate_nack_transaction()
    animate_transaction(nack_seg, "I2C Error: NACK (Device Not Found)")

if __name__ == "__main__":
    main()