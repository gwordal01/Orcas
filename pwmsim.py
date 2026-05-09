"""
Minimal PWM Simulator
=====================
Interactive PWM simulator with:
- Slider to control duty cycle (0–100%)
- Real-time square wave visualization
- Average voltage display
- Virtual LED brightness proportional to duty cycle

Dependencies:
    pip install matplotlib numpy
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# ── Config ───────────────────────────────────────────────
V_HIGH   = 5.0       # High voltage (like Arduino 5V)
V_LOW    = 0.0       # Low voltage
FREQ_HZ  = 2.0       # Frequency (Hz)
CYCLES   = 3         # How many cycles to show
POINTS   = 1000      # Resolution
INIT_DUTY = 50.0     # Initial duty cycle %

# ── Waveform generation ──────────────────────────────────
def make_pwm_wave(duty_pct, freq=FREQ_HZ, cycles=CYCLES, n_points=POINTS):
    t_total = cycles / freq
    t = np.linspace(0, t_total, n_points, endpoint=False)
    period = 1.0 / freq
    phase = (t % period) / period
    duty = duty_pct / 100.0
    v = np.where(phase < duty, V_HIGH, V_LOW)
    return t, v

def average_voltage(duty_pct):
    return (duty_pct / 100.0) * V_HIGH

# ── Figure setup ─────────────────────────────────────────
fig, (ax_wave, ax_led) = plt.subplots(2, 1, figsize=(8, 6))
plt.subplots_adjust(bottom=0.25)

# Waveform plot
t_init, v_init = make_pwm_wave(INIT_DUTY)
wave_line, = ax_wave.plot(t_init, v_init, lw=2)
ax_wave.set_ylim(-0.5, V_HIGH + 0.5)
ax_wave.set_xlim(0, CYCLES / FREQ_HZ)
ax_wave.set_xlabel("Time (s)")
ax_wave.set_ylabel("Voltage (V)")
ax_wave.set_title("PWM Square Wave")

# Average voltage line
avg_line = ax_wave.axhline(average_voltage(INIT_DUTY), color="orange", ls="--")
avg_text = ax_wave.text(0.02, average_voltage(INIT_DUTY)+0.2,
                        f"V_avg = {average_voltage(INIT_DUTY):.2f} V",
                        color="orange")

# LED plot
ax_led.set_xlim(-1, 1)
ax_led.set_ylim(-1, 1)
ax_led.set_aspect("equal")
ax_led.axis("off")
led_circle = plt.Circle((0, 0), 0.4, color="yellow", alpha=INIT_DUTY/100.0)
ax_led.add_patch(led_circle)
ax_led.set_title("Virtual LED")

# Slider
ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])
slider = Slider(ax_slider, "Duty Cycle (%)", 0, 100, valinit=INIT_DUTY)

# ── Update function ──────────────────────────────────────
def update(val):
    duty = slider.val
    t, v = make_pwm_wave(duty)
    wave_line.set_data(t, v)
    avg_v = average_voltage(duty)
    avg_line.set_ydata([avg_v, avg_v])
    avg_text.set_position((0.02, avg_v + 0.2))
    avg_text.set_text(f"V_avg = {avg_v:.2f} V")
    led_circle.set_alpha(duty/100.0)
    fig.canvas.draw_idle()

slider.on_changed(update)

plt.show()
