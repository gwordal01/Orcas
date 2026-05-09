from pathlib import Path
import threading
from datetime import datetime

import cv2
import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    ort = None

try:
    import torch
except ImportError:
    torch = None


SCRIPT_DIR = Path(__file__).resolve().parent
ONNX_PATH = SCRIPT_DIR / "midas_small.onnx"
WINDOW_NAME = "DepthMapper"
EXPORT_STEP = 6
CENTER_BOX_SIZE = 80
MAX_DISPLAY_WIDTH = 1280


class ONNXBackend:
    def __init__(self, model_path: Path):
        if ort is None:
            raise RuntimeError("onnxruntime is not installed")
        providers = ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(str(model_path), providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        _, _, self.input_h, self.input_w = self.session.get_inputs()[0].shape
        self.input_h = int(self.input_h)
        self.input_w = int(self.input_w)
        print(f"Loading ONNX MiDaS-small from {model_path} ...")
        print("Model ready (ONNX Runtime).")

    def estimate(self, bgr_frame: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (self.input_w, self.input_h), interpolation=cv2.INTER_CUBIC)
        inp = resized.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        inp = (inp - mean) / std
        inp = np.transpose(inp, (2, 0, 1))[None, ...]

        pred = self.session.run(None, {self.input_name: inp})[0][0]
        depth = cv2.resize(pred, (bgr_frame.shape[1], bgr_frame.shape[0]), interpolation=cv2.INTER_CUBIC)
        return normalize_depth(depth)


class MiDaSBackend:
    def __init__(self, device: str = "cpu"):
        if torch is None:
            raise RuntimeError("torch is not installed")
        self.device = torch.device(device)
        print("Loading MiDaS-small model from torch hub ...")
        self.model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True)
        transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
        self.transform = transforms.small_transform
        self.model.to(self.device).eval()
        print("Model ready (PyTorch).")

    def estimate(self, bgr_frame: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        inp = self.transform(rgb).to(self.device)
        with torch.no_grad():
            pred = self.model(inp)
            pred = torch.nn.functional.interpolate(
                pred.unsqueeze(1),
                size=bgr_frame.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        depth = pred.detach().cpu().numpy()
        return normalize_depth(depth)


def normalize_depth(depth: np.ndarray) -> np.ndarray:
    depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    d_min = float(depth.min())
    d_max = float(depth.max())
    if d_max - d_min < 1e-6:
        return np.zeros_like(depth, dtype=np.float32)
    return (depth - d_min) / (d_max - d_min)


def create_backend():
    if ONNX_PATH.exists() and ort is not None:
        return ONNXBackend(ONNX_PATH)
    return MiDaSBackend(device="cpu")


def estimate_center_region(depth_map: np.ndarray, box_size: int = CENTER_BOX_SIZE):
    h, w = depth_map.shape
    half = box_size // 2
    cx, cy = w // 2, h // 2
    x0 = max(cx - half, 0)
    x1 = min(cx + half, w)
    y0 = max(cy - half, 0)
    y1 = min(cy + half, h)
    region = depth_map[y0:y1, x0:x1]
    center_depth = float(np.median(region)) if region.size else float(depth_map[cy, cx])

    relative_distance_mm = int(np.interp(center_depth, [0.0, 1.0], [4000, 300]))
    return (x0, y0, x1, y1), center_depth, relative_distance_mm


def export_point_cloud(depth_map: np.ndarray, output_path: Path, step: int = EXPORT_STEP):
    h, w = depth_map.shape
    ys, xs = np.mgrid[0:h:step, 0:w:step]
    ds = depth_map[::step, ::step]
    data = np.column_stack([xs.ravel(), ys.ravel(), ds.ravel()])
    np.savetxt(
        output_path,
        data,
        delimiter=",",
        header="x,y,depth",
        comments="",
        fmt=["%d", "%d", "%.6f"],
    )
    print(f"Saved point cloud with {len(data)} points to {output_path}")


class ExportManager:
    def __init__(self):
        self.thread = None
        self.last_csv_path = None
        self.last_csv_error = None
        self.last_screenshot_path = None
        self.in_progress = False

    def start_export(self, depth_map: np.ndarray):
        if self.in_progress:
            return False, self.last_csv_path

        output_path = SCRIPT_DIR / f"point_cloud_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        depth_copy = depth_map.copy()
        self.in_progress = True
        self.last_csv_error = None
        self.last_csv_path = output_path

        def worker():
            try:
                export_point_cloud(depth_copy, output_path)
            except Exception as exc:
                self.last_csv_error = str(exc)
            finally:
                self.in_progress = False

        self.thread = threading.Thread(target=worker, daemon=True)
        self.thread.start()
        return True, output_path


def draw_histogram(depth_map: np.ndarray, width: int, height: int) -> np.ndarray:
    hist_canvas = np.zeros((height, width, 3), dtype=np.uint8)
    counts, _ = np.histogram(depth_map.ravel(), bins=60, range=(0.0, 1.0))
    max_count = max(int(counts.max()), 1)
    bar_w = max(width // len(counts), 1)

    for idx, count in enumerate(counts):
        bar_h = int((count / max_count) * (height - 30))
        x0 = idx * bar_w
        x1 = min(x0 + bar_w - 1, width - 1)
        cv2.rectangle(hist_canvas, (x0, height - 1), (x1, height - bar_h), (0, 215, 255), -1)

    cv2.putText(hist_canvas, "Depth Histogram", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(hist_canvas, "far", (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1, cv2.LINE_AA)
    cv2.putText(hist_canvas, "near", (width - 52, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1, cv2.LINE_AA)
    return hist_canvas


def fit_width(image: np.ndarray, max_width: int) -> np.ndarray:
    h, w = image.shape[:2]
    if w <= max_width:
        return image
    scale = max_width / float(w)
    return cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def build_canvas(frame: np.ndarray, depth_map: np.ndarray, exporter: ExportManager, fps: float) -> np.ndarray:
    display_frame = frame.copy()
    box, center_depth, distance_mm = estimate_center_region(depth_map)
    x0, y0, x1, y1 = box
    cv2.rectangle(display_frame, (x0, y0), (x1, y1), (80, 255, 80), 2)
    cv2.putText(display_frame, "Center region", (x0, max(25, y0 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 255, 80), 2, cv2.LINE_AA)

    depth_uint8 = np.clip(depth_map * 255.0, 0, 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_TURBO)
    cv2.rectangle(heatmap, (x0, y0), (x1, y1), (255, 255, 255), 2)

    hist = draw_histogram(depth_map, frame.shape[1], 220)
    top = np.hstack([display_frame, heatmap])
    canvas = np.vstack([top, np.hstack([hist, np.zeros_like(hist)])])

    cv2.putText(canvas, f"FPS: {fps:.1f}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, f"Center depth: {center_depth:.3f}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, f"Estimated distance: {distance_mm} mm (relative)", (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, "[Space] export CSV  [S] save screenshot  [Q/ESC] quit", (20, canvas.shape[0] - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (220, 220, 220), 2, cv2.LINE_AA)

    if exporter.in_progress:
        cv2.putText(canvas, "Saving CSV...", (frame.shape[1] + 20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 255), 2, cv2.LINE_AA)
    elif exporter.last_csv_error:
        cv2.putText(canvas, f"CSV error: {exporter.last_csv_error}", (frame.shape[1] + 20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
    elif exporter.last_csv_path is not None:
        cv2.putText(canvas, f"CSV saved: {exporter.last_csv_path.name}", (frame.shape[1] + 20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(canvas, str(exporter.last_csv_path.parent), (frame.shape[1] + 20, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

    if exporter.last_screenshot_path is not None:
        cv2.putText(canvas, f"Shot: {exporter.last_screenshot_path.name}", (frame.shape[1] + 20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 220, 120), 2, cv2.LINE_AA)

    cv2.putText(canvas, "Camera", (20, frame.shape[0] + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, "Depth Heatmap", (frame.shape[1] + 20, frame.shape[0] + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

    return fit_width(canvas, MAX_DISPLAY_WIDTH)


def main():
    backend = create_backend()
    exporter = ExportManager()
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 1280, 900)

    last_tick = cv2.getTickCount()
    fps = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        depth_map = backend.estimate(frame)
        now = cv2.getTickCount()
        dt = (now - last_tick) / cv2.getTickFrequency()
        if dt > 0:
            fps = 1.0 / dt
        last_tick = now

        canvas = build_canvas(frame, depth_map, exporter, fps)
        cv2.imshow(WINDOW_NAME, canvas)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        if key == ord(' '):
            exporter.start_export(depth_map)
        if key == ord('s'):
            screenshot_path = SCRIPT_DIR / f"depthmapper_frame_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            cv2.imwrite(str(screenshot_path), canvas)
            exporter.last_screenshot_path = screenshot_path

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
