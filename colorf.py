#!/usr/bin/env python3
"""
ObjectFollower Day 23
=====================
Track a colored object, compute pixel error from frame center, and emit a PID
style control signal suitable for a future pan-tilt servo loop.

Controls:
    S           Enter sampling mode, then click the target color
    R           Reset target and controller state
    M           Toggle mask inset
    T           Toggle trail
    [ / ]       Decrease / increase hue tolerance
    - / =       Decrease / increase minimum contour area
    1 / 2       Decrease / increase Kp
    3 / 4       Decrease / increase Ki
    5 / 6       Decrease / increase Kd
    0           Zero Ki and Kd quickly
    C           Save a snapshot
    Q / ESC     Quit
"""

import argparse
import csv
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


DEFAULT_CAMERA = 0
DEFAULT_KP = 0.40
DEFAULT_KI = 0.015
DEFAULT_KD = 0.12
DEFAULT_MIN_AREA = 700
DEFAULT_HUE_TOL = 16
DEFAULT_SAT_MIN = 70
DEFAULT_VAL_MIN = 45
MAX_CONTROL = 100.0
INTEGRAL_LIMIT = 4000.0
DERIVATIVE_ALPHA = 0.22
TRAIL_LENGTH = 40
MISS_LIMIT = 12
DEADBAND_PX = 4.0

HUD_BG = (16, 16, 16)
TEXT_COLOR = (235, 235, 235)
SUBTLE_TEXT = (160, 160, 160)
CENTER_COLOR = (0, 220, 255)
TARGET_COLOR = (60, 240, 80)
WARNING_COLOR = (0, 140, 255)
ACCENT_X = (70, 170, 255)
ACCENT_Y = (0, 255, 180)

RESOLUTION_LADDER = [
    (1920, 1080),
    (1280, 720),
    (960, 540),
    (640, 480),
]

WINDOW_NAME = "ObjectFollower Day 23"
CLICK_STATE = {"pending": False, "x": 0, "y": 0}


@dataclass
class AxisTelemetry:
    output: float = 0.0
    p: float = 0.0
    i: float = 0.0
    d: float = 0.0


class PIDController:
    def __init__(self, kp, ki, kd, integral_limit=INTEGRAL_LIMIT):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral_limit = integral_limit
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_time = None
        self.filtered_derivative = 0.0

    def update(self, error):
        now = time.perf_counter()
        if self.prev_time is None:
            dt = 1.0 / 60.0
        else:
            dt = max(1e-3, min(0.25, now - self.prev_time))

        if abs(error) < DEADBAND_PX:
            error = 0.0

        p_term = self.kp * error

        self.integral += error * dt
        self.integral = float(
            np.clip(self.integral, -self.integral_limit, self.integral_limit)
        )
        i_term = self.ki * self.integral

        derivative = (error - self.prev_error) / dt
        self.filtered_derivative = (
            DERIVATIVE_ALPHA * derivative
            + (1.0 - DERIVATIVE_ALPHA) * self.filtered_derivative
        )
        d_term = self.kd * self.filtered_derivative

        output = float(np.clip(p_term + i_term + d_term, -MAX_CONTROL, MAX_CONTROL))

        self.prev_error = error
        self.prev_time = now
        return AxisTelemetry(output=output, p=p_term, i=i_term, d=d_term)

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_time = None
        self.filtered_derivative = 0.0

    def set_gains(self, kp=None, ki=None, kd=None):
        if kp is not None:
            self.kp = max(0.0, kp)
        if ki is not None:
            self.ki = max(0.0, ki)
        if kd is not None:
            self.kd = max(0.0, kd)


def open_camera(index):
    cap = cv2.VideoCapture(index, cv2.CAP_ANY)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera {index}")

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    accepted_w = accepted_h = 0
    for width, height in RESOLUTION_LADDER:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        got_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        got_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        accepted_w, accepted_h = got_w, got_h
        if got_w >= width * 0.9 and got_h >= height * 0.9:
            break

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    print(f"[INFO] Camera {index} opened at {accepted_w}x{accepted_h} @ {fps:.0f} fps")
    return cap


def circular_mean_hue(hues):
    hue_values = hues.flatten().astype(np.float32)
    angles = hue_values * (2.0 * np.pi / 180.0)
    sin_mean = np.mean(np.sin(angles))
    cos_mean = np.mean(np.cos(angles))
    angle = np.arctan2(sin_mean, cos_mean)
    if angle < 0:
        angle += 2.0 * np.pi
    return int(round(angle * 180.0 / (2.0 * np.pi))) % 180


def build_mask(hsv_frame, hue, tol, sat_min, val_min):
    lo_h = (hue - tol) % 180
    hi_h = (hue + tol) % 180

    if lo_h <= hi_h:
        mask = cv2.inRange(
            hsv_frame,
            np.array([lo_h, sat_min, val_min]),
            np.array([hi_h, 255, 255]),
        )
    else:
        mask = (
            cv2.inRange(
                hsv_frame,
                np.array([lo_h, sat_min, val_min]),
                np.array([179, 255, 255]),
            )
            | cv2.inRange(
                hsv_frame,
                np.array([0, sat_min, val_min]),
                np.array([hi_h, 255, 255]),
            )
        )

    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    return mask


def largest_contour(mask, min_area):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    return contour if cv2.contourArea(contour) >= min_area else None


def contour_centroid(contour):
    moments = cv2.moments(contour)
    if moments["m00"] == 0:
        return None
    return (
        int(moments["m10"] / moments["m00"]),
        int(moments["m01"] / moments["m00"]),
    )


def draw_crosshair(frame, cx, cy, size=18, color=CENTER_COLOR, thickness=2):
    cv2.line(frame, (cx - size, cy), (cx + size, cy), color, thickness)
    cv2.line(frame, (cx, cy - size), (cx, cy + size), color, thickness)
    cv2.circle(frame, (cx, cy), size // 2, color, thickness)


def draw_vector_scope(frame, center, target):
    cv2.arrowedLine(frame, center, target, (40, 210, 90), 2, tipLength=0.14)
    cv2.circle(frame, target, 6, (40, 210, 90), -1)


def draw_pid_bar(frame, x, y, width, label, value, color, max_value=MAX_CONTROL):
    cv2.rectangle(frame, (x, y), (x + width, y + 16), (70, 70, 70), 1)
    midpoint = x + width // 2
    cv2.line(frame, (midpoint, y), (midpoint, y + 16), (100, 100, 100), 1)
    fill = int(min(abs(value), max_value) / max_value * (width // 2))
    if value >= 0:
        cv2.rectangle(frame, (midpoint, y + 1), (midpoint + fill, y + 15), color, -1)
    else:
        cv2.rectangle(frame, (midpoint - fill, y + 1), (midpoint, y + 15), color, -1)
    cv2.putText(
        frame,
        f"{label}: {value:+6.2f}",
        (x, y - 6),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.44,
        TEXT_COLOR,
        1,
        cv2.LINE_AA,
    )


def draw_trail(frame, points):
    if len(points) < 2:
        return
    for idx in range(1, len(points)):
        if points[idx - 1] is None or points[idx] is None:
            continue
        alpha = idx / len(points)
        color = (
            int(40 + 180 * alpha),
            int(80 + 120 * alpha),
            int(255 * (1.0 - alpha * 0.45)),
        )
        cv2.line(frame, points[idx - 1], points[idx], color, 2)


def save_snapshot(frame):
    path = Path(f"objectfollower_snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    cv2.imwrite(str(path), frame)
    print(f"[INFO] Snapshot saved to {path}")


def on_mouse(event, x, y, flags, state):
    if event == cv2.EVENT_LBUTTONDOWN and state["sampling"]:
        CLICK_STATE["pending"] = True
        CLICK_STATE["x"] = x
        CLICK_STATE["y"] = y


def sample_target(frame, state):
    frame_h, frame_w = frame.shape[:2]
    px, py = CLICK_STATE["x"], CLICK_STATE["y"]
    radius = 6
    y0, y1 = max(0, py - radius), min(frame_h, py + radius + 1)
    x0, x1 = max(0, px - radius), min(frame_w, px + radius + 1)
    patch_bgr = frame[y0:y1, x0:x1]
    if patch_bgr.size == 0:
        return

    patch_hsv = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2HSV)
    hue = circular_mean_hue(patch_hsv[:, :, 0])
    sat_median = int(np.median(patch_hsv[:, :, 1]))
    val_median = int(np.median(patch_hsv[:, :, 2]))

    center_y = patch_bgr.shape[0] // 2
    center_x = patch_bgr.shape[1] // 2
    state["sampled_bgr"] = patch_bgr[center_y, center_x].tolist()
    state["hue"] = hue
    state["sat_min"] = max(DEFAULT_SAT_MIN, int(sat_median * 0.55))
    state["val_min"] = max(DEFAULT_VAL_MIN, int(val_median * 0.55))
    state["sampling"] = False
    state["lost_frames"] = 0
    state["trail"].clear()

    print(
        "[INFO] Sampled target:",
        f"hue={state['hue']}",
        f"sat_min={state['sat_min']}",
        f"val_min={state['val_min']}",
        f"at=({px},{py})",
    )


def draw_hud(frame, state):
    frame_h, frame_w = frame.shape[:2]
    center_x, center_y = frame_w // 2, frame_h // 2
    draw_crosshair(frame, center_x, center_y, size=14, color=CENTER_COLOR, thickness=1)

    overlay = frame.copy()
    hud_height = 150
    cv2.rectangle(overlay, (0, frame_h - hud_height), (frame_w, frame_h), HUD_BG, -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)

    top = frame_h - hud_height + 24
    cv2.putText(
        frame,
        f"Tracking: {'ON' if state['tracking'] else 'SEARCH'}",
        (16, top),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        TARGET_COLOR if state["tracking"] else WARNING_COLOR,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        (
            f"Hue {state['hue'] if state['hue'] is not None else '--'}  "
            f"Tol +/-{state['tol']}  "
            f"Area>{state['min_area']} px^2"
        ),
        (16, top + 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        SUBTLE_TEXT,
        1,
        cv2.LINE_AA,
    )

    if state["tracking"]:
        ex, ey = state["error"]
        obj_x, obj_y = state["obj_center"]
        norm_x = ex / max(1, frame_w // 2)
        norm_y = ey / max(1, frame_h // 2)

        cv2.putText(
            frame,
            f"Object ({obj_x:4d}, {obj_y:4d})  Error dx={ex:+7.1f} dy={ey:+7.1f} px",
            (16, top + 52),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            TEXT_COLOR,
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            (
                f"Normalized dx={norm_x:+.3f} dy={norm_y:+.3f}  "
                f"Confidence={state['confidence']:.2f}"
            ),
            (16, top + 74),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            TEXT_COLOR,
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            (
                f"Pan  P={state['pan_terms'].p:+6.2f} I={state['pan_terms'].i:+6.2f} "
                f"D={state['pan_terms'].d:+6.2f} Out={state['pan_terms'].output:+6.2f}"
            ),
            (16, top + 98),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            ACCENT_X,
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            (
                f"Tilt P={state['tilt_terms'].p:+6.2f} I={state['tilt_terms'].i:+6.2f} "
                f"D={state['tilt_terms'].d:+6.2f} Out={state['tilt_terms'].output:+6.2f}"
            ),
            (16, top + 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            ACCENT_Y,
            1,
            cv2.LINE_AA,
        )
    else:
        cv2.putText(
            frame,
            "S sample color  |  click target  |  1/2 Kp  3/4 Ki  5/6 Kd  |  C snapshot",
            (16, top + 52),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.44,
            SUBTLE_TEXT,
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            "Use HSV sampling to reject the background. Largest contour wins.",
            (16, top + 76),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.44,
            SUBTLE_TEXT,
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            (
                f"Gains Kp={state['pid_x'].kp:.3f} Ki={state['pid_x'].ki:.3f} "
                f"Kd={state['pid_x'].kd:.3f}"
            ),
            (16, top + 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            TEXT_COLOR,
            1,
            cv2.LINE_AA,
        )

    draw_pid_bar(frame, frame_w - 310, frame_h - 96, 280, "PAN", state["pan_terms"].output, ACCENT_X)
    draw_pid_bar(frame, frame_w - 310, frame_h - 56, 280, "TILT", state["tilt_terms"].output, ACCENT_Y)

    cv2.putText(
        frame,
        f"{state['fps']:.0f} fps",
        (frame_w - 95, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        SUBTLE_TEXT,
        1,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        (
            f"Gains Kp={state['pid_x'].kp:.3f} Ki={state['pid_x'].ki:.3f} "
            f"Kd={state['pid_x'].kd:.3f}"
        ),
        (frame_w - 310, frame_h - 118),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.44,
        TEXT_COLOR,
        1,
        cv2.LINE_AA,
    )

    if state["sampling"]:
        cv2.rectangle(frame, (0, 0), (frame_w - 1, frame_h - 1), WARNING_COLOR, 4)
        prompt = "Click the target color"
        text_w = cv2.getTextSize(prompt, cv2.FONT_HERSHEY_SIMPLEX, 0.72, 2)[0][0]
        cv2.putText(
            frame,
            prompt,
            ((frame_w - text_w) // 2, 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            WARNING_COLOR,
            2,
            cv2.LINE_AA,
        )

    if state["hue"] is not None:
        swatch = np.zeros((22, 58, 3), dtype=np.uint8)
        swatch[:] = state["sampled_bgr"] if state["sampled_bgr"] is not None else (0, 0, 0)
        frame[30:52, frame_w - 70:frame_w - 12] = swatch
        cv2.rectangle(frame, (frame_w - 70, 30), (frame_w - 12, 52), (190, 190, 190), 1)


def adjust_gain(pid_x, pid_y, field, delta):
    current = getattr(pid_x, field)
    updated = max(0.0, current + delta)
    pid_x.set_gains(**{field: updated})
    pid_y.set_gains(**{field: updated})
    print(f"[INFO] {field.upper()} set to {updated:.3f}")


def run(camera_idx, kp, ki, kd, min_area, record):
    cap = open_camera(camera_idx)
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    pid_x = PIDController(kp, ki, kd)
    pid_y = PIDController(kp, ki, kd)

    state = {
        "hue": None,
        "sampled_bgr": None,
        "tol": DEFAULT_HUE_TOL,
        "sat_min": DEFAULT_SAT_MIN,
        "val_min": DEFAULT_VAL_MIN,
        "min_area": min_area,
        "sampling": False,
        "tracking": False,
        "show_mask": True,
        "show_trail": True,
        "fps": 0.0,
        "error": (0.0, 0.0),
        "obj_center": (0, 0),
        "confidence": 0.0,
        "pan_terms": AxisTelemetry(),
        "tilt_terms": AxisTelemetry(),
        "pid_x": pid_x,
        "pid_y": pid_y,
        "trail": deque(maxlen=TRAIL_LENGTH),
        "lost_frames": 0,
    }

    cv2.setMouseCallback(WINDOW_NAME, on_mouse, state)

    log_file = None
    log_writer = None
    if record:
        log_path = Path(f"tracking_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        log_file = open(log_path, "w", newline="", encoding="utf-8")
        log_writer = csv.writer(log_file)
        log_writer.writerow(
            [
                "timestamp",
                "obj_x",
                "obj_y",
                "error_x",
                "error_y",
                "pan_output",
                "tilt_output",
                "pan_p",
                "pan_i",
                "pan_d",
                "tilt_p",
                "tilt_i",
                "tilt_d",
                "confidence",
            ]
        )
        print(f"[INFO] Logging to {log_path}")

    print("[INFO] S -> sample target color, then click the object.")
    print("[INFO] 1/2 Kp, 3/4 Ki, 5/6 Kd, [/] hue tol, -/= min area, M mask, T trail.")

    prev_frame_time = time.perf_counter()
    smooth_center = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Camera frame read failed. Retrying...")
            time.sleep(0.03)
            continue

        frame_h, frame_w = frame.shape[:2]
        frame_center = (frame_w // 2, frame_h // 2)

        now = time.perf_counter()
        dt = max(1e-6, now - prev_frame_time)
        state["fps"] = 1.0 / dt
        prev_frame_time = now

        if CLICK_STATE["pending"] and state["sampling"]:
            CLICK_STATE["pending"] = False
            sample_target(frame, state)
            pid_x.reset()
            pid_y.reset()
            smooth_center = None

        display = frame.copy()
        state["tracking"] = False
        state["confidence"] = 0.0
        state["pan_terms"] = AxisTelemetry()
        state["tilt_terms"] = AxisTelemetry()

        if state["hue"] is not None:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = build_mask(
                hsv, state["hue"], state["tol"], state["sat_min"], state["val_min"]
            )
            contour = largest_contour(mask, state["min_area"])

            if contour is not None:
                center = contour_centroid(contour)
                if center is not None:
                    area = cv2.contourArea(contour)
                    hull = cv2.convexHull(contour)
                    hull_area = max(cv2.contourArea(hull), 1.0)
                    confidence = float(np.clip(area / hull_area, 0.0, 1.0))

                    if smooth_center is None:
                        smooth_center = np.array(center, dtype=np.float32)
                    else:
                        smooth_center = 0.65 * smooth_center + 0.35 * np.array(center, dtype=np.float32)

                    cx_obj, cy_obj = int(smooth_center[0]), int(smooth_center[1])
                    error_x = float(cx_obj - frame_center[0])
                    error_y = float(cy_obj - frame_center[1])

                    pan_terms = pid_x.update(error_x)
                    tilt_terms = pid_y.update(error_y)

                    state["tracking"] = True
                    state["lost_frames"] = 0
                    state["error"] = (error_x, error_y)
                    state["obj_center"] = (cx_obj, cy_obj)
                    state["confidence"] = confidence
                    state["pan_terms"] = pan_terms
                    state["tilt_terms"] = tilt_terms
                    state["trail"].append((cx_obj, cy_obj))

                    cv2.drawContours(display, [contour], -1, TARGET_COLOR, 2)
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.rectangle(display, (x, y), (x + w, y + h), (0, 200, 80), 1)
                    draw_crosshair(display, cx_obj, cy_obj, size=20, color=TARGET_COLOR, thickness=2)
                    draw_vector_scope(display, frame_center, (cx_obj, cy_obj))

                    if log_writer:
                        log_writer.writerow(
                            [
                                f"{time.time():.4f}",
                                cx_obj,
                                cy_obj,
                                f"{error_x:.2f}",
                                f"{error_y:.2f}",
                                f"{pan_terms.output:.4f}",
                                f"{tilt_terms.output:.4f}",
                                f"{pan_terms.p:.4f}",
                                f"{pan_terms.i:.4f}",
                                f"{pan_terms.d:.4f}",
                                f"{tilt_terms.p:.4f}",
                                f"{tilt_terms.i:.4f}",
                                f"{tilt_terms.d:.4f}",
                                f"{confidence:.3f}",
                            ]
                        )
                else:
                    state["lost_frames"] += 1
            else:
                state["lost_frames"] += 1

            if state["lost_frames"] > 0:
                state["trail"].append(None)

            if state["lost_frames"] >= MISS_LIMIT:
                pid_x.reset()
                pid_y.reset()
                smooth_center = None
                state["pan_terms"] = AxisTelemetry()
                state["tilt_terms"] = AxisTelemetry()

            if state["show_mask"]:
                inset_w = min(240, frame_w // 4)
                inset_h = min(180, frame_h // 4)
                mask_small = cv2.resize(mask, (inset_w, inset_h))
                mask_bgr = cv2.cvtColor(mask_small, cv2.COLOR_GRAY2BGR)
                mask_bgr[mask_small > 0] = [40, 210, 90]
                display[8:8 + inset_h, 8:8 + inset_w] = mask_bgr
                cv2.rectangle(display, (8, 8), (8 + inset_w, 8 + inset_h), (100, 100, 100), 1)
                cv2.putText(
                    display,
                    "HSV mask",
                    (12, 8 + inset_h + 16),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    SUBTLE_TEXT,
                    1,
                    cv2.LINE_AA,
                )

        if state["show_trail"]:
            draw_trail(display, list(state["trail"]))

        draw_hud(display, state)
        cv2.imshow(WINDOW_NAME, display)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q"), 27):
            break
        if key in (ord("s"), ord("S")):
            state["sampling"] = True
            print("[INFO] Sampling mode active. Click the target.")
        elif key in (ord("r"), ord("R")):
            state["hue"] = None
            state["sampled_bgr"] = None
            state["sat_min"] = DEFAULT_SAT_MIN
            state["val_min"] = DEFAULT_VAL_MIN
            state["tracking"] = False
            state["sampling"] = False
            state["trail"].clear()
            state["lost_frames"] = 0
            pid_x.reset()
            pid_y.reset()
            smooth_center = None
            print("[INFO] Reset target and PID state.")
        elif key == ord("m") or key == ord("M"):
            state["show_mask"] = not state["show_mask"]
            print(f"[INFO] Show mask: {state['show_mask']}")
        elif key == ord("t") or key == ord("T"):
            state["show_trail"] = not state["show_trail"]
            print(f"[INFO] Show trail: {state['show_trail']}")
        elif key == ord("["):
            state["tol"] = max(4, state["tol"] - 2)
            print(f"[INFO] Hue tolerance +/-{state['tol']}")
        elif key == ord("]"):
            state["tol"] = min(60, state["tol"] + 2)
            print(f"[INFO] Hue tolerance +/-{state['tol']}")
        elif key == ord("-"):
            state["min_area"] = max(100, state["min_area"] - 50)
            print(f"[INFO] Min contour area {state['min_area']}")
        elif key in (ord("="), ord("+")):
            state["min_area"] = min(10000, state["min_area"] + 50)
            print(f"[INFO] Min contour area {state['min_area']}")
        elif key == ord("1"):
            adjust_gain(pid_x, pid_y, "kp", -0.02)
        elif key == ord("2"):
            adjust_gain(pid_x, pid_y, "kp", 0.02)
        elif key == ord("3"):
            adjust_gain(pid_x, pid_y, "ki", -0.005)
        elif key == ord("4"):
            adjust_gain(pid_x, pid_y, "ki", 0.005)
        elif key == ord("5"):
            adjust_gain(pid_x, pid_y, "kd", -0.01)
        elif key == ord("6"):
            adjust_gain(pid_x, pid_y, "kd", 0.01)
        elif key == ord("0"):
            adjust_gain(pid_x, pid_y, "ki", -pid_x.ki)
            adjust_gain(pid_x, pid_y, "kd", -pid_x.kd)
        elif key in (ord("c"), ord("C")):
            save_snapshot(display)

    cap.release()
    cv2.destroyAllWindows()
    if log_file:
        log_file.close()
        print("[INFO] Tracking log saved.")


def main():
    parser = argparse.ArgumentParser(description="Day 23 ObjectFollower")
    parser.add_argument("--camera", type=int, default=DEFAULT_CAMERA, help="Camera index")
    parser.add_argument("--kp", type=float, default=DEFAULT_KP, help="Proportional gain")
    parser.add_argument("--ki", type=float, default=DEFAULT_KI, help="Integral gain")
    parser.add_argument("--kd", type=float, default=DEFAULT_KD, help="Derivative gain")
    parser.add_argument(
        "--min-area",
        type=int,
        default=DEFAULT_MIN_AREA,
        help="Minimum contour area in pixels",
    )
    parser.add_argument("--record", action="store_true", help="Write CSV telemetry log")
    args = parser.parse_args()

    try:
        run(args.camera, args.kp, args.ki, args.kd, args.min_area, args.record)
    except RuntimeError as exc:
        print(f"[ERROR] {exc}")


if __name__ == "__main__":
    main()
