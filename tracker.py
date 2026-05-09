"""
╔══════════════════════════════════════════════════════════════════╗
║          OBJECT FOLLOWER — Pan-Tilt P-Controller Core           ║
║          Day 23 · Robotics Vision Pipeline · v1.0               ║
╚══════════════════════════════════════════════════════════════════╝

Architecture:
  Camera → HSV Threshold → Contour Detection → Centroid →
  Error Signal → P-Controller → [servo output in v2.0]

The (dx, dy) error IS the P-term that drives a real pan-tilt servo.
"""

import cv2
import numpy as np
import json
import time
import argparse
from dataclasses import dataclass, asdict
from collections import deque
from datetime import datetime
import os

# ─── Theme Colors (BGR for OpenCV) ───────────────────────────────
PINK        = (180, 60, 255)    # glowy dark pink
PINK_LIGHT  = (210, 130, 255)
PINK_DARK   = (120, 20, 180)
CYAN        = (255, 220, 0)
WHITE       = (255, 255, 255)
BLACK       = (0, 0, 0)
DARK_BG     = (20, 10, 30)
GREEN_OK    = (80, 220, 120)
RED_ERR     = (60, 60, 220)


@dataclass
class ControlSignal:
    """Proportional controller output — ready for servo consumption."""
    timestamp: float
    frame_idx: int

    # Raw pixel error (P-term input)
    error_x: float          # pixels left(-) / right(+) of center
    error_y: float          # pixels up(-) / down(+) of center

    # Normalized error [-1, 1]
    norm_x: float
    norm_y: float

    # P-controller output (what a servo PWM loop receives)
    p_out_x: float          # pan  signal
    p_out_y: float          # tilt signal

    # Object stats
    area: float
    centroid_x: int
    centroid_y: int
    confidence: float       # 0-1 contour solidity

    # Tracking state
    object_found: bool


class KalmanTracker:
    """2-D Kalman filter for smooth centroid tracking."""
    def __init__(self):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix  = np.array([[1,0,0,0],[0,1,0,0]], np.float32)
        self.kf.transitionMatrix   = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]], np.float32)
        self.kf.processNoiseCov    = np.eye(4, dtype=np.float32) * 0.03
        self.kf.measurementNoiseCov= np.eye(2, dtype=np.float32) * 1.0
        self.initialized = False

    def update(self, cx, cy):
        measurement = np.array([[np.float32(cx)], [np.float32(cy)]])
        if not self.initialized:
            self.kf.statePre = np.array([[cx],[cy],[0],[0]], np.float32)
            self.initialized = True
        self.kf.correct(measurement)
        predicted = self.kf.predict()
        return int(predicted[0]), int(predicted[1])

    def predict_only(self):
        if not self.initialized:
            return None
        predicted = self.kf.predict()
        return int(predicted[0]), int(predicted[1])


class HSVColorPicker:
    """Interactive HSV range picker — click to sample target color."""
    def __init__(self):
        self.picking = False
        self.hsv_lower = np.array([0, 100, 100])
        self.hsv_upper = np.array([10, 255, 255])
        self.sample_point = None
        self.sample_radius = 15
        self.history = []          # recent picked colors for averaging

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.sample_point = (x, y)
            self.picking = True

    def sample_from_frame(self, hsv_frame, bgr_frame):
        if not self.picking or self.sample_point is None:
            return
        x, y = self.sample_point
        h, w = hsv_frame.shape[:2]
        r = self.sample_radius

        # Sample a patch around click
        x1, y1 = max(0, x-r), max(0, y-r)
        x2, y2 = min(w, x+r), min(h, y+r)
        patch = hsv_frame[y1:y2, x1:x2]

        if patch.size == 0:
            self.picking = False
            return

        # Robust: use median not mean
        h_med = np.median(patch[:,:,0])
        s_med = np.median(patch[:,:,1])
        v_med = np.median(patch[:,:,2])

        h_std = np.std(patch[:,:,0])
        s_std = np.std(patch[:,:,1])

        h_range = max(15, int(h_std * 2.5))
        s_range = max(40, int(s_std * 2.5))

        self.hsv_lower = np.array([
            max(0,   h_med - h_range),
            max(0,   s_med - s_range),
            max(30,  v_med - 60)
        ], dtype=np.uint8)
        self.hsv_upper = np.array([
            min(179, h_med + h_range),
            min(255, s_med + s_range),
            min(255, v_med + 60)
        ], dtype=np.uint8)

        self.history.append((h_med, s_med, v_med))
        self.picking = False
        print(f"[PICKER] HSV lower={self.hsv_lower}  upper={self.hsv_upper}")


class ObjectFollower:
    def __init__(self, camera_id=0, Kp=0.5, min_area=800, log_signals=False):
        self.camera_id   = camera_id
        self.Kp          = Kp          # Proportional gain
        self.min_area    = min_area    # Minimum contour area (noise filter)
        self.log_signals = log_signals

        self.cap         = None
        self.frame_idx   = 0
        self.tracker     = KalmanTracker()
        self.picker      = HSVColorPicker()

        # Telemetry history (circular buffer)
        self.history     = deque(maxlen=300)   # ~10s at 30fps
        self.error_trail = deque(maxlen=60)    # centroid trail

        # Stats
        self.fps_buffer  = deque(maxlen=30)
        self.last_time   = time.time()
        self.lost_frames = 0
        self.found_frames= 0

        # Log file
        if log_signals:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.log_path = f"logs/signals_{ts}.jsonl"
            os.makedirs("logs", exist_ok=True)
        else:
            self.log_path = None

    # ──────────────────────────── Core Processing ────────────────

    def detect_object(self, frame):
        """Find the largest colored contour → centroid + stats."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Sample color if user clicked
        self.picker.sample_from_frame(hsv, frame)

        # Threshold
        lo, hi = self.picker.hsv_lower, self.picker.hsv_upper

        # Handle hue wrap-around (e.g. red: 170-10)
        if lo[0] > hi[0]:
            mask1 = cv2.inRange(hsv, lo, np.array([179, hi[1], hi[2]]))
            mask2 = cv2.inRange(hsv, np.array([0, lo[1], lo[2]]), hi)
            mask  = cv2.bitwise_or(mask1, mask2)
        else:
            mask  = cv2.inRange(hsv, lo, hi)

        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
        mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
        mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask   = cv2.GaussianBlur(mask, (5,5), 0)
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None, mask

        # Keep only largest contour (filter noise)
        largest = max(contours, key=cv2.contourArea)
        area    = cv2.contourArea(largest)

        if area < self.min_area:
            return None, mask

        # Centroid via moments
        M  = cv2.moments(largest)
        if M["m00"] == 0:
            return None, mask
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        # Solidity = contour area / convex hull area (shape quality metric)
        hull      = cv2.convexHull(largest)
        hull_area = cv2.contourArea(hull)
        solidity  = float(area) / hull_area if hull_area > 0 else 0

        return {
            "cx": cx, "cy": cy,
            "area": area,
            "contour": largest,
            "solidity": solidity,
            "mask": mask
        }, mask

    def compute_control(self, cx, cy, frame_w, frame_h):
        """P-controller: error → normalized signal → servo output."""
        center_x = frame_w / 2
        center_y = frame_h / 2

        # Raw pixel error
        err_x = cx - center_x   # +right, -left
        err_y = cy - center_y   # +down,  -up

        # Normalize to [-1, 1]
        norm_x = err_x / center_x
        norm_y = err_y / center_y

        # Proportional output (clamped — real servo safety)
        p_x = float(np.clip(self.Kp * norm_x, -1.0, 1.0))
        p_y = float(np.clip(self.Kp * norm_y, -1.0, 1.0))

        return err_x, err_y, norm_x, norm_y, p_x, p_y

    # ──────────────────────────── Rendering ──────────────────────

    def draw_reticle(self, frame, cx, cy, size=30, color=PINK):
        """Animated targeting reticle around centroid."""
        t = time.time()
        pulse = int(5 * abs(np.sin(t * 4)))     # pulsing size

        # Corner brackets
        for sx, sy in [(-1,-1),(-1,1),(1,-1),(1,1)]:
            x0 = cx + sx * (size + pulse)
            y0 = cy + sy * (size + pulse)
            cv2.line(frame, (x0, y0), (x0 + sx*-15, y0), color, 2)
            cv2.line(frame, (x0, y0), (x0, y0 + sy*-15), color, 2)

        # Center crosshair
        cv2.line(frame, (cx-8, cy), (cx+8, cy), color, 1)
        cv2.line(frame, (cx, cy-8), (cx, cy+8), color, 1)
        cv2.circle(frame, (cx, cy), 3, color, -1)

    def draw_error_vector(self, frame, cx, cy, frame_w, frame_h, err_x, err_y):
        """Arrow from frame center to object centroid."""
        fc_x, fc_y = frame_w//2, frame_h//2
        # Scale arrow
        scale = 0.4
        ax = int(fc_x + err_x * scale)
        ay = int(fc_y + err_y * scale)
        cv2.arrowedLine(frame, (fc_x, fc_y), (ax, ay), PINK_LIGHT, 2, tipLength=0.3)

    def draw_trail(self, frame):
        """Centroid motion trail."""
        pts = list(self.error_trail)
        for i in range(1, len(pts)):
            alpha = i / len(pts)
            color = (
                int(PINK_DARK[0] * alpha + DARK_BG[0] * (1-alpha)),
                int(PINK_DARK[1] * alpha + DARK_BG[1] * (1-alpha)),
                int(PINK_DARK[2] * alpha + DARK_BG[2] * (1-alpha)),
            )
            thickness = max(1, int(alpha * 3))
            cv2.line(frame, pts[i-1], pts[i], color, thickness)

    def draw_crosshair_center(self, frame, w, h):
        """Frame center marker."""
        cx, cy = w//2, h//2
        cv2.line(frame, (cx-20, cy), (cx+20, cy), (80,80,80), 1)
        cv2.line(frame, (cx, cy-20), (cx, cy+20), (80,80,80), 1)
        cv2.circle(frame, (cx, cy), 4, PINK_DARK, 1)

    def draw_hud(self, frame, signal: ControlSignal, fps, w, h):
        """Full overlay HUD — telemetry panel."""
        # Semi-transparent dark panel (left side)
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (300, h), (15, 8, 25), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

        # Title
        cv2.putText(frame, "OBJECT FOLLOWER", (12, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, PINK, 2)
        cv2.putText(frame, "v1.0  P-CONTROLLER", (12, 52),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, PINK_DARK, 1)

        # Divider
        cv2.line(frame, (12, 62), (288, 62), PINK_DARK, 1)

        def label_val(y, label, val, color=WHITE, alert=False):
            col = (60,60,220) if alert else color
            cv2.putText(frame, label, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160,100,180), 1)
            cv2.putText(frame, str(val), (150, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1)

        # Status
        status     = "LOCKED" if signal.object_found else "SEARCHING"
        status_col = GREEN_OK if signal.object_found else RED_ERR
        cv2.putText(frame, f"STATUS: {status}", (12, 84),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, status_col, 2)

        label_val(110, "FPS",         f"{fps:.1f}")
        label_val(128, "FRAME",       f"#{signal.frame_idx}")
        label_val(146, "AREA (px²)",  f"{int(signal.area)}")
        label_val(164, "SOLIDITY",    f"{signal.confidence:.2f}")

        cv2.line(frame, (12, 176), (288, 176), PINK_DARK, 1)
        cv2.putText(frame, "ERROR  (pixels)", (12, 194),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, PINK_LIGHT, 1)

        ex_alert = abs(signal.error_x) > w * 0.3
        ey_alert = abs(signal.error_y) > h * 0.3
        label_val(212, "err_x (pan)",  f"{signal.error_x:+.1f} px", alert=ex_alert)
        label_val(230, "err_y (tilt)", f"{signal.error_y:+.1f} px", alert=ey_alert)
        label_val(248, "norm_x",       f"{signal.norm_x:+.3f}")
        label_val(266, "norm_y",       f"{signal.norm_y:+.3f}")

        cv2.line(frame, (12, 278), (288, 278), PINK_DARK, 1)
        cv2.putText(frame, f"P-CONTROLLER  Kp={self.Kp}", (12, 296),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, PINK_LIGHT, 1)

        label_val(314, "p_out_x (pan)",  f"{signal.p_out_x:+.4f}")
        label_val(332, "p_out_y (tilt)", f"{signal.p_out_y:+.4f}")

        # Mini bar graphs for p_out
        self._draw_bar(frame, 12, 344, 276, 14, signal.p_out_x, "PAN ",  PINK)
        self._draw_bar(frame, 12, 362, 276, 14, signal.p_out_y, "TILT", CYAN)

        # HSV picker info
        cv2.line(frame, (12, 382), (288, 382), PINK_DARK, 1)
        lo, hi = self.picker.hsv_lower, self.picker.hsv_upper
        cv2.putText(frame, f"HSV  [{lo[0]},{lo[1]},{lo[2]}]→[{hi[0]},{hi[1]},{hi[2]}]",
                    (12, 398), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (130,80,160), 1)

        # Bottom hint
        hints = "[C]lick=pick color  [R]eset  [+/-]Kp  [Q]uit"
        cv2.putText(frame, hints, (12, h-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (100,60,130), 1)

        # Session stats (bottom right)
        total = self.found_frames + self.lost_frames
        pct   = 100 * self.found_frames / total if total else 0
        cv2.putText(frame, f"LOCK RATE  {pct:.0f}%",
                    (w-160, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.38, PINK_DARK, 1)

    def _draw_bar(self, frame, x, y, w, h, value, label, color):
        """Horizontal bar graph for controller output [-1, 1]."""
        mid  = x + w // 2
        bw   = int(abs(value) * (w // 2))
        bx   = mid if value >= 0 else mid - bw

        cv2.rectangle(frame, (x, y), (x+w, y+h), (40,20,50), -1)
        cv2.rectangle(frame, (bx, y+2), (bx+bw, y+h-2), color, -1)
        cv2.line(frame, (mid, y), (mid, y+h), (80,40,90), 1)

        cv2.putText(frame, label, (x+2, y+10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, WHITE, 1)
        cv2.putText(frame, f"{value:+.3f}", (x+w-60, y+10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, WHITE, 1)

    def draw_scope(self, frame, w, h):
        """Mini oscilloscope: error_x / error_y history."""
        scope_x, scope_y = w - 220, 20
        scope_w, scope_h = 200, 100
        overlay = frame.copy()
        cv2.rectangle(overlay, (scope_x, scope_y),
                      (scope_x+scope_w, scope_y+scope_h), (15,8,25), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        cv2.rectangle(frame, (scope_x, scope_y),
                      (scope_x+scope_w, scope_y+scope_h), PINK_DARK, 1)
        cv2.putText(frame, "SCOPE", (scope_x+4, scope_y+12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, PINK_DARK, 1)

        # Draw x error (pink) and y error (cyan)
        history = list(self.history)[-scope_w:]
        if len(history) < 2:
            return
        mid_y = scope_y + scope_h // 2
        for ch, color, attr in [(0, PINK, "norm_x"), (1, CYAN, "norm_y")]:
            pts = []
            for i, sig in enumerate(history):
                val = getattr(sig, attr) if hasattr(sig, attr) else 0
                px  = scope_x + int(i * scope_w / len(history))
                py  = mid_y   - int(val * (scope_h//2 - 4))
                py  = max(scope_y+2, min(scope_y+scope_h-2, py))
                pts.append((px, py))
            for i in range(1, len(pts)):
                cv2.line(frame, pts[i-1], pts[i], color, 1)

        # Center line
        cv2.line(frame, (scope_x, mid_y), (scope_x+scope_w, mid_y), (60,30,70), 1)

    # ──────────────────────────── Main Loop ──────────────────────

    def run(self):
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            print(f"[ERROR] Cannot open camera {self.camera_id}")
            return

        # Prefer HD
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        win_name = "ObjectFollower — P-Controller"
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win_name, 1280, 720)
        cv2.setMouseCallback(win_name, self.picker.mouse_callback)

        mask_win = "HSV Mask"
        cv2.namedWindow(mask_win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(mask_win, 320, 180)

        print("\n╔══════════════════════════════════╗")
        print("║   OBJECT FOLLOWER  starting...   ║")
        print("╠══════════════════════════════════╣")
        print("║  Click on object to set color    ║")
        print("║  +/-  adjust Kp gain             ║")
        print("║  R    reset tracker              ║")
        print("║  S    save snapshot              ║")
        print("║  Q    quit                       ║")
        print("╚══════════════════════════════════╝\n")

        log_file = open(self.log_path, "w") if self.log_path else None

        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("[WARN] Frame grab failed — retrying...")
                time.sleep(0.05)
                continue

            self.frame_idx += 1
            h, w = frame.shape[:2]

            # FPS
            now  = time.time()
            self.fps_buffer.append(1.0 / max(now - self.last_time, 1e-6))
            self.last_time = now
            fps = np.mean(self.fps_buffer)

            # ── Detection ────────────────────────────────────────
            detection, mask = self.detect_object(frame)

            if detection:
                # Kalman smooth
                cx, cy = self.tracker.update(detection["cx"], detection["cy"])
                self.error_trail.append((cx, cy))
                self.found_frames += 1

                err_x, err_y, nx, ny, px, py = self.compute_control(cx, cy, w, h)

                signal = ControlSignal(
                    timestamp   = now,
                    frame_idx   = self.frame_idx,
                    error_x     = err_x,
                    error_y     = err_y,
                    norm_x      = nx,
                    norm_y      = ny,
                    p_out_x     = px,
                    p_out_y     = py,
                    area        = detection["area"],
                    centroid_x  = cx,
                    centroid_y  = cy,
                    confidence  = detection["solidity"],
                    object_found= True
                )

                # Draw detection visuals
                cv2.drawContours(frame, [detection["contour"]], -1, PINK_DARK, 2)
                self.draw_trail(frame)
                self.draw_error_vector(frame, cx, cy, w, h, err_x, err_y)
                self.draw_reticle(frame, cx, cy, color=PINK)

                # Bounding box
                x,y,bw,bh = cv2.boundingRect(detection["contour"])
                cv2.rectangle(frame, (x,y), (x+bw,y+bh), PINK_DARK, 1)

                # Centroid label
                cv2.putText(frame, f"({cx},{cy})", (cx+12, cy-12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, PINK_LIGHT, 1)

            else:
                # Predict position even without detection
                pred = self.tracker.predict_only()
                self.lost_frames += 1

                null_xy = pred if pred else (w//2, h//2)
                signal = ControlSignal(
                    timestamp=now, frame_idx=self.frame_idx,
                    error_x=0, error_y=0, norm_x=0, norm_y=0,
                    p_out_x=0, p_out_y=0,
                    area=0, centroid_x=null_xy[0], centroid_y=null_xy[1],
                    confidence=0, object_found=False
                )

                if pred:
                    # Ghost reticle at predicted position
                    self.draw_reticle(frame, pred[0], pred[1],
                                      size=25, color=PINK_DARK)
                    cv2.putText(frame, "PREDICTING", (pred[0]+14, pred[1]-14),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, PINK_DARK, 1)

            self.history.append(signal)

            # ── HUD ──────────────────────────────────────────────
            self.draw_crosshair_center(frame, w, h)
            self.draw_hud(frame, signal, fps, w, h)
            self.draw_scope(frame, w, h)

            # ── Mask window ───────────────────────────────────────
            mask_color = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            mask_small = cv2.resize(mask_color, (320, 180))
            cv2.putText(mask_small, "HSV MASK", (4, 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, PINK, 1)
            cv2.imshow(mask_win, mask_small)

            # ── Log ──────────────────────────────────────────────
            if log_file and signal.object_found:
                log_file.write(json.dumps(asdict(signal)) + "\n")

            # ── Display ───────────────────────────────────────────
            cv2.imshow(win_name, frame)

            # ── Key handling ──────────────────────────────────────
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                self.tracker = KalmanTracker()
                self.error_trail.clear()
                print("[INFO] Tracker reset")
            elif key == ord('+') or key == ord('='):
                self.Kp = min(2.0, round(self.Kp + 0.05, 2))
                print(f"[Kp] → {self.Kp}")
            elif key == ord('-'):
                self.Kp = max(0.05, round(self.Kp - 0.05, 2))
                print(f"[Kp] → {self.Kp}")
            elif key == ord('s'):
                ts_str = datetime.now().strftime("%H%M%S")
                fname  = f"snapshot_{ts_str}.jpg"
                cv2.imwrite(fname, frame)
                print(f"[SNAP] Saved {fname}")

        self.cap.release()
        cv2.destroyAllWindows()
        if log_file:
            log_file.close()
            print(f"[LOG] Signals saved → {self.log_path}")

        total = self.found_frames + self.lost_frames
        print(f"\n[SESSION] Frames: {total}  |  Lock rate: "
              f"{100*self.found_frames/max(total,1):.1f}%  |  FPS avg: {fps:.1f}")


# ─── CLI ──────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="ObjectFollower P-Controller")
    ap.add_argument("--camera",  type=int,   default=0,    help="Camera index")
    ap.add_argument("--kp",      type=float, default=0.5,  help="Proportional gain")
    ap.add_argument("--minarea", type=int,   default=800,  help="Minimum contour area")
    ap.add_argument("--log",     action="store_true",      help="Log signals to JSONL")
    args = ap.parse_args()

    follower = ObjectFollower(
        camera_id  = args.camera,
        Kp         = args.kp,
        min_area   = args.minarea,
        log_signals= args.log
    )
    follower.run()

if __name__ == "__main__":
    main()