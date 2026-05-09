import numpy as np
import sounddevice as sd
import pygame
import threading

# --- AUDIO ENGINE CONFIG ---
SAMPLE_RATE = 44100
BUFFER_SIZE = 512
MAX_VOICES = 4
MASTER_GAIN = 0.12  # Adjusted for 4-voice headroom

# Echo / Delay System
ECHO_DELAY_MS = 300
ECHO_FEEDBACK = 0.35
ECHO_SAMPLES = int((ECHO_DELAY_MS / 1000) * SAMPLE_RATE)
echo_buffer = np.zeros(SAMPLE_RATE * 3) 
echo_ptr = 0

active_notes = {} # {key: {"freq": f, "phase": p}}
echo_enabled = False
lock = threading.Lock()

def audio_callback(outdata, frames, time_info, status):
    global echo_ptr
    t = np.arange(frames) / SAMPLE_RATE
    mixed = np.zeros(frames)
    
    with lock:
        for key, data in list(active_notes.items()):
            freq = data["freq"]
            wave = np.sin(2 * np.pi * freq * t + data["phase"])
            mixed += wave * MASTER_GAIN
            data["phase"] = (data["phase"] + 2 * np.pi * freq * frames / SAMPLE_RATE) % (2 * np.pi)
            
    if echo_enabled:
        for i in range(frames):
            delayed_idx = (echo_ptr - ECHO_SAMPLES) % len(echo_buffer)
            delayed_sample = echo_buffer[delayed_idx]
            out_sample = mixed[i] + (delayed_sample * ECHO_FEEDBACK)
            echo_buffer[echo_ptr] = mixed[i] + (delayed_sample * ECHO_FEEDBACK)
            mixed[i] = out_sample
            echo_ptr = (echo_ptr + 1) % len(echo_buffer)

    outdata[:, 0] = np.tanh(mixed).astype(np.float32)

# --- FULL ALPHABET MAPPING (C4 to C6) ---
# Bottom row: Z-M | Home row: A-L | Top row: Q-P
NOTES = {
    # Lower Octave
    pygame.K_z: 261.63, pygame.K_x: 293.66, pygame.K_c: 329.63, pygame.K_v: 349.23,
    pygame.K_b: 392.00, pygame.K_n: 440.00, pygame.K_m: 493.88,
    # Middle Octave
    pygame.K_a: 523.25, pygame.K_s: 587.33, pygame.K_d: 659.25, pygame.K_f: 698.46,
    pygame.K_g: 783.99, pygame.K_h: 880.00, pygame.K_j: 987.77, pygame.K_k: 1046.50,
    pygame.K_l: 1108.73,
    # High Octave
    pygame.K_q: 1174.66, pygame.K_w: 1318.51, pygame.K_e: 1396.91, pygame.K_r: 1567.98,
    pygame.K_t: 1760.00, pygame.K_y: 1975.53, pygame.K_u: 2093.00, pygame.K_i: 2349.32,
    pygame.K_o: 2637.02, pygame.K_p: 2793.83
}

# --- UI DESIGN ---
pygame.init()
WIDTH, HEIGHT = 1100, 650
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("ORCA Master Station | Day 07")

CLR_BG = (248, 249, 252)
CLR_MAIN = (0, 110, 255)
CLR_DARK = (30, 32, 40)
CLR_TEXT_SEC = (150, 155, 170)

FONT_MAIN = pygame.font.SysFont("Arial", 16)
FONT_BOLD = pygame.font.SysFont("Arial", 20, bold=True)

def draw_ui(surface):
    # Main Waveform Window
    screen_rect = pygame.Rect(80, 100, 940, 300)
    pygame.draw.rect(surface, (255, 255, 255), screen_rect, border_radius=20)
    pygame.draw.rect(surface, (230, 235, 245), screen_rect, 2, border_radius=20)

    with lock:
        freq_vals = [d["freq"] for d in active_notes.values()]
    
    # Superposition Visualization
    if freq_vals:
        t_draw = np.linspace(0, 0.02, 800)
        combined = np.zeros(800)
        for f in freq_vals:
            combined += np.sin(2 * np.pi * f * t_draw)
        combined = np.tanh(combined * 0.5)
        
        pts = []
        for i, v in enumerate(combined):
            x = 80 + (i * 940 // 800)
            y = 250 - (v * 100)
            pts.append((x, y))
        if len(pts) > 1:
            pygame.draw.aalines(surface, CLR_MAIN, False, pts)

    # Info Bar
    echo_status = "ECHO ENABLED" if echo_enabled else "ECHO MUTED"
    echo_color = CLR_MAIN if echo_enabled else CLR_TEXT_SEC
    surface.blit(FONT_BOLD.render(echo_status, True, echo_color), (80, 420))
    
    usage = FONT_MAIN.render(f"ACTIVE VOICES: {len(freq_vals)} / 4", True, CLR_DARK)
    surface.blit(usage, (WIDTH - 80 - usage.get_width(), 420))

    # Instructions Panel
    y_start = 520
    instructions = [
        "PLAY: Z-M (Low) | A-L (Mid) | Q-P (High)",
        "CONTROLS: [SPACE] Toggle Echo  |  [ESC] Exit Program",
        "NOTE: Max 4 simultaneous tones for harmonic clarity"
    ]
    for i, line in enumerate(instructions):
        txt = FONT_MAIN.render(line, True, CLR_TEXT_SEC if i==2 else CLR_DARK)
        surface.blit(txt, (WIDTH//2 - txt.get_width()//2, y_start + (i*24)))

# --- MAIN LOOP ---
stream = sd.OutputStream(channels=1, callback=audio_callback, samplerate=SAMPLE_RATE)
stream.start()
clock = pygame.time.Clock()
running = True

while running:
    screen.fill(CLR_BG)
    for event in pygame.event.get():
        if event.type == pygame.QUIT: running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE: running = False
            if event.key == pygame.K_SPACE: echo_enabled = not echo_enabled
            if event.key in NOTES and event.key not in active_notes:
                with lock:
                    if len(active_notes) < MAX_VOICES:
                        active_notes[event.key] = {"freq": NOTES[event.key], "phase": 0}
        if event.type == pygame.KEYUP:
            if event.key in NOTES:
                with lock: active_notes.pop(event.key, None)

    draw_ui(screen)
    pygame.display.flip()
    clock.tick(60)

stream.stop()
pygame.quit()
