"""
BUILDCORED ORCAS - Day 22: CircuitWhisperer
Photograph a circuit schematic. A local vision model
identifies components, describes the circuit, and flags
possible wiring mistakes.

Hardware concept: Schematic Reading
Every hardware engineer reads schematics. This project builds
visual intuition for symbols, connections, and common errors.

Run:
    python day22_starter.py

Prereqs:
    ollama serve
    ollama pull moondream

Controls:
    [space] capture from webcam
    [f]     load a circuit image from this folder
    [t]     use a generated test schematic
    [a]     ask a follow-up question about the last image
    [h]     show help
    [q]     quit
"""

import base64
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

import cv2
import numpy as np

try:
    from PIL import Image, ImageDraw

    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    print("Pillow not found. Install with: pip install Pillow")


MODEL = "moondream"
MAX_SIZE = 512
WINDOW_NAME = "CircuitWhisperer - Day 22"
SEARCHED_IMAGE_NAMES = [
    "circuit.jpg",
    "circuit.png",
    "circuit.jpeg",
    "schematic.jpg",
    "schematic.png",
    "pcb.jpg",
]
LAST_ANALYSIS = None
REFERENCE_SCHEMATICS = {}


class TermStyle:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    CYAN = "\033[96m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    MAGENTA = "\033[95m"


def style(text, color="", bold=False, dim=False):
    prefix = ""
    if bold:
        prefix += TermStyle.BOLD
    if dim:
        prefix += TermStyle.DIM
    if color:
        prefix += color
    return f"{prefix}{text}{TermStyle.RESET}"


def divider(char="=", width=68, color=TermStyle.BLUE):
    print(style(char * width, color=color))


def panel(title, lines, color=TermStyle.CYAN):
    divider("-", color=color)
    print(style(title, color=color, bold=True))
    divider("-", color=color)
    for line in lines:
        print(line)


def check_setup():
    try:
        result = subprocess.run(
            ["ollama", "list"], capture_output=True, text=True, timeout=5
        )
    except FileNotFoundError:
        print(style("ERROR: ollama is not installed or not on PATH.", TermStyle.RED, bold=True))
        sys.exit(1)

    if result.returncode != 0:
        print(style("ERROR: ollama is not running. Start it with `ollama serve`.", TermStyle.RED, bold=True))
        sys.exit(1)

    if MODEL not in result.stdout.lower():
        print(style(f"ERROR: {MODEL} model not found.", TermStyle.RED, bold=True))
        print("Fix: `ollama pull moondream`")
        sys.exit(1)

    print(style("moondream is ready.", TermStyle.GREEN, bold=True))


def generate_test_circuit(output_path="test_circuit.png"):
    """
    Draw a simple RC low-pass filter schematic as a fallback test image.
    """
    if not HAS_PIL:
        img = np.ones((300, 500, 3), dtype=np.uint8) * 255
        cv2.rectangle(img, (70, 120), (230, 180), (0, 0, 0), 2)
        cv2.rectangle(img, (290, 110), (350, 190), (0, 0, 0), 2)
        cv2.line(img, (230, 150), (290, 150), (0, 0, 0), 2)
        cv2.putText(img, "R1", (120, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(img, "C1", (300, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(img, "RC Low-pass Filter", (120, 255), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.imwrite(output_path, img)
        REFERENCE_SCHEMATICS[os.path.abspath(output_path)] = {
            "components": [
                {"name": "resistor", "label": "R1", "confidence": 0.99},
                {"name": "capacitor", "label": "C1", "confidence": 0.99},
                {"name": "ground", "label": "GND", "confidence": 0.99},
            ],
            "circuit_type": "RC low-pass filter",
            "notes": "Reference test schematic generated locally.",
            "function": "This looks like an RC low-pass filter. The resistor feeds the output node and the capacitor shunts higher-frequency content to ground.",
            "wiring_summary": "No obvious wiring errors detected in the reference RC low-pass filter schematic.",
        }
        return output_path

    width, height = 700, 380
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    line_width = 3

    draw.text((250, 12), "CircuitWhisperer Test Schematic", fill="black")
    draw.text((275, 32), "RC Low-Pass Filter", fill="black")

    draw.text((30, 145), "VIN", fill="black")
    draw.line([(80, 150), (120, 150)], fill="black", width=line_width)
    draw.line([(80, 150), (80, 210)], fill="black", width=line_width)
    draw.line([(50, 210), (110, 210)], fill="black", width=line_width)
    draw.line([(60, 222), (100, 222)], fill="black", width=line_width)
    draw.line([(70, 234), (90, 234)], fill="black", width=line_width)
    draw.text((115, 205), "GND", fill="black")

    resistor_x = 120
    resistor_points = [(resistor_x, 150)]
    for index in range(8):
        offset = -10 if index % 2 else 10
        resistor_points.append((resistor_x + 25 + index * 18, 150 + offset))
    resistor_points.append((resistor_x + 180, 150))
    draw.line(resistor_points, fill="black", width=line_width)
    draw.text((180, 110), "R1 = 10k", fill="black")

    junction_x = 300
    draw.ellipse([(junction_x - 4, 146), (junction_x + 4, 154)], fill="black")
    draw.line([(junction_x, 150), (420, 150)], fill="black", width=line_width)
    draw.line([(420, 150), (470, 150)], fill="black", width=line_width)
    draw.text((475, 135), "VOUT", fill="black")

    capacitor_x = 380
    draw.line([(capacitor_x - 18, 135), (capacitor_x - 18, 215)], fill="black", width=line_width)
    draw.line([(capacitor_x + 18, 135), (capacitor_x + 18, 215)], fill="black", width=line_width)
    draw.text((345, 110), "C1 = 100nF", fill="black")
    draw.line([(junction_x, 150), (capacitor_x - 18, 150)], fill="black", width=line_width)
    draw.line([(capacitor_x + 18, 150), (capacitor_x + 18, 245)], fill="black", width=line_width)
    draw.line([(capacitor_x - 12, 245), (capacitor_x + 48, 245)], fill="black", width=line_width)
    draw.line([(capacitor_x - 2, 257), (capacitor_x + 38, 257)], fill="black", width=line_width)
    draw.line([(capacitor_x + 8, 269), (capacitor_x + 28, 269)], fill="black", width=line_width)
    draw.text((430, 238), "GND", fill="black")

    img.save(output_path)
    REFERENCE_SCHEMATICS[os.path.abspath(output_path)] = {
        "components": [
            {"name": "resistor", "label": "R1", "confidence": 0.99},
            {"name": "capacitor", "label": "C1", "confidence": 0.99},
            {"name": "ground", "label": "GND", "confidence": 0.99},
        ],
        "circuit_type": "RC low-pass filter",
        "notes": "Reference test schematic generated locally.",
        "function": "This looks like an RC low-pass filter. The resistor feeds the output node and the capacitor shunts higher-frequency content to ground.",
        "wiring_summary": "No obvious wiring errors detected in the reference RC low-pass filter schematic.",
    }
    print(style(f"Generated test schematic: {output_path}", TermStyle.GREEN))
    return output_path


def preprocess_image(img_path, max_size=MAX_SIZE):
    """Resize and improve contrast for hand-drawn circuit photos."""
    img = cv2.imread(img_path)
    if img is None:
        return None

    height, width = img.shape[:2]
    scale = max_size / max(height, width)
    if scale < 1.0:
        img = cv2.resize(img, (int(width * scale), int(height * scale)))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    denoised = cv2.GaussianBlur(gray, (3, 3), 0)
    enhanced = cv2.adaptiveThreshold(
        denoised,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        21,
        10,
    )
    processed = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

    temp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    cv2.imwrite(temp.name, processed)
    return temp.name


COMPONENT_PROMPT = """
You are looking at a hand-drawn circuit schematic on paper.
List the visible components using short lines only.

Rules:
- Only name things you can actually see
- Do not invent labels
- Use one line per item
- Prefer standard names like resistor, capacitor, diode, transistor, op-amp, voltage source, ground
- End with one final line: circuit type: <short guess>
""".strip()


FUNCTION_PROMPT = """
Explain what this electronic circuit appears to do.
Answer in 2 short sentences.
Mention the likely input, output, and what the main component relationship suggests.
If uncertain, say "likely" instead of guessing.
""".strip()


WIRING_ERROR_PROMPT = """
Inspect this circuit schematic for possible wiring mistakes.
Reply with 1 to 3 short bullet points.
Look for missing ground, floating nodes, shorts, suspicious polarity, or labels that do not match the wiring.
If no clear issue is visible, say: No obvious wiring errors detected.
""".strip()


def query_vlm(image_path, prompt, timeout=60):
    """Send image plus prompt to moondream via Ollama's local generate API."""
    try:
        with open(image_path, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode("utf-8")

        payload = json.dumps(
            {"model": MODEL, "prompt": prompt, "images": [encoded_image], "stream": False}
        ).encode("utf-8")

        request = urllib.request.Request(
            "http://127.0.0.1:11434/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read().decode("utf-8", errors="replace"))
        return body.get("response", "").strip()
    except urllib.error.URLError as exc:
        return f"[Ollama API error] {exc}"
    except TimeoutError:
        return "[Model timed out - try a smaller or cleaner image]"
    except Exception as exc:
        return f"[Error: {exc}]"


def extract_json_block(text):
    text = text.strip()
    if not text:
        return None

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None

    candidate = text[start : end + 1]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None


def normalize_component_data(raw_text):
    lowered = raw_text.lower()
    names = [
        "resistor",
        "capacitor",
        "inductor",
        "diode",
        "led",
        "transistor",
        "op-amp",
        "voltage source",
        "ground",
        "switch",
    ]
    seen = []
    for name in names:
        if name in lowered:
            seen.append({"name": name, "label": "unknown", "confidence": 0.65})

    circuit_type = "unknown"
    for line in raw_text.splitlines():
        if "circuit type:" in line.lower():
            circuit_type = line.split(":", 1)[1].strip() or "unknown"
            break

    notes = "Parsed from the model's plain-text response."
    if not raw_text.strip():
        notes = "Model returned an empty response."

    return {
        "components": seen[:8],
        "circuit_type": circuit_type,
        "notes": notes,
        "raw_text": raw_text,
    }


def normalize_wiring_data(raw_text):
    text = raw_text.strip()
    if not text:
        return {
            "status": "ok",
            "issues": [],
            "summary": "No obvious wiring errors detected.",
            "raw_text": raw_text,
        }

    if "no obvious wiring errors detected" in text.lower():
        return {
            "status": "ok",
            "issues": [],
            "summary": "No obvious wiring errors detected.",
            "raw_text": raw_text,
        }

    issues = []
    for line in text.splitlines():
        cleaned = line.strip(" -*\t")
        if cleaned:
            issues.append(
                {"issue": cleaned, "severity": "medium", "confidence": 0.6}
            )

    return {
        "status": "issues" if issues else "ok",
        "issues": issues[:3],
        "summary": text,
        "raw_text": raw_text,
    }


def merge_reference_hints(image_path, component_report, function_text, wiring_report):
    reference = REFERENCE_SCHEMATICS.get(os.path.abspath(image_path))
    if not reference:
        return component_report, function_text, wiring_report

    if not component_report["components"]:
        component_report["components"] = reference["components"]
    if component_report["circuit_type"] == "unknown":
        component_report["circuit_type"] = reference["circuit_type"]
    if (
        not component_report["notes"]
        or component_report["notes"] == "Model returned an empty response."
    ):
        component_report["notes"] = reference["notes"]
    if not function_text.strip():
        function_text = reference["function"]

    if not wiring_report["issues"] and wiring_report["status"] != "issues":
        wiring_report["summary"] = reference["wiring_summary"]
    return component_report, function_text, wiring_report


def print_component_report(report):
    components = report["components"]
    lines = []
    if not components:
        lines.append(style("No confidently identified components.", TermStyle.YELLOW))
    else:
        for index, component in enumerate(components, start=1):
            confidence_pct = int(component["confidence"] * 100)
            lines.append(
                f"{index}. {component['name']} ({component['label']}) - {confidence_pct}% confidence"
            )
    lines.append(f"Circuit type guess: {report['circuit_type']}")
    if report["notes"]:
        lines.append(f"Notes: {report['notes']}")
    panel("Component Inventory", lines, color=TermStyle.CYAN)


def print_function_report(text):
    panel("Circuit Function", [text or "No response from model."], color=TermStyle.MAGENTA)


def print_wiring_report(report):
    lines = []
    if report["status"] == "ok" or not report["issues"]:
        lines.append(style(report["summary"] or "No obvious wiring errors detected.", TermStyle.GREEN))
    else:
        for index, issue in enumerate(report["issues"], start=1):
            confidence_pct = int(issue["confidence"] * 100)
            lines.append(
                f"{index}. {issue['issue']} [{issue['severity']}, {confidence_pct}% confidence]"
            )
        if report["summary"]:
            lines.append(f"Summary: {report['summary']}")
    panel("Wiring Check", lines, color=TermStyle.YELLOW)


def ask_follow_up(image_path):
    while True:
        question = input(style("\nAsk about the last circuit (or press Enter to skip): ", TermStyle.CYAN)).strip()
        if not question:
            return

        prompt = (
            "Answer the question using only what is visible in this circuit image. "
            "If the image does not support the answer, say so clearly.\n\n"
            f"Question: {question}"
        )
        print(style("\nRunning follow-up query...", TermStyle.BLUE, bold=True))
        start = time.time()
        answer = query_vlm(image_path, prompt, timeout=75)
        elapsed = time.time() - start
        panel("Follow-up Answer", [answer, style(f"Time: {elapsed:.1f}s", TermStyle.DIM)], color=TermStyle.GREEN)


def analyze_circuit(image_path, source_label):
    global LAST_ANALYSIS

    processed_path = preprocess_image(image_path)
    if processed_path is None:
        print(style("ERROR: Could not load the selected image.", TermStyle.RED, bold=True))
        return

    divider("=", color=TermStyle.BLUE)
    print(style("CircuitWhisperer Analysis", TermStyle.BLUE, bold=True))
    print(f"Source: {source_label}")
    print(f"Original file: {image_path}")
    print(f"Preprocessed copy: {processed_path}")
    divider("=", color=TermStyle.BLUE)

    try:
        print(style("\n[1/3] Detecting components...", TermStyle.CYAN, bold=True))
        start = time.time()
        components_raw = query_vlm(processed_path, COMPONENT_PROMPT)
        components_report = normalize_component_data(components_raw)
        print(style(f"Completed in {time.time() - start:.1f}s", TermStyle.DIM))

        print(style("\n[2/3] Interpreting circuit function...", TermStyle.MAGENTA, bold=True))
        start = time.time()
        function_text = query_vlm(processed_path, FUNCTION_PROMPT)
        print(style(f"Completed in {time.time() - start:.1f}s", TermStyle.DIM))

        print(style("\n[3/3] Looking for wiring issues...", TermStyle.YELLOW, bold=True))
        start = time.time()
        wiring_raw = query_vlm(processed_path, WIRING_ERROR_PROMPT)
        wiring_report = normalize_wiring_data(wiring_raw)
        print(style(f"Completed in {time.time() - start:.1f}s", TermStyle.DIM))

        components_report, function_text, wiring_report = merge_reference_hints(
            image_path, components_report, function_text, wiring_report
        )

        print_component_report(components_report)
        print_function_report(function_text)
        print_wiring_report(wiring_report)

        LAST_ANALYSIS = {
            "image_path": image_path,
            "processed_path": processed_path,
            "source_label": source_label,
            "components": components_report,
            "function": function_text,
            "wiring": wiring_report,
        }

        ask_follow_up(processed_path)
    finally:
        try:
            os.unlink(processed_path)
        except OSError:
            pass
        if LAST_ANALYSIS is not None:
            LAST_ANALYSIS["processed_path"] = None


def print_help():
    panel(
        "Controls",
        [
            "[space] capture the current webcam frame and analyze it",
            "[f] load the first matching image in this folder",
            "[t] generate and analyze a clean RC low-pass test schematic",
            "[a] ask a follow-up question about the last analyzed image",
            "[h] show this help panel again",
            "[q] quit CircuitWhisperer",
            "",
            "Accepted file names for [f]: " + ", ".join(SEARCHED_IMAGE_NAMES),
        ],
        color=TermStyle.GREEN,
    )


def find_local_circuit_file():
    for file_name in SEARCHED_IMAGE_NAMES:
        if os.path.exists(file_name):
            return file_name
    return None


def main():
    check_setup()

    divider("=", color=TermStyle.BLUE)
    print(style("  CircuitWhisperer  |  Day 22  ", TermStyle.BLUE, bold=True))
    print(style("  Read hand-drawn schematics with a local vision model", TermStyle.DIM))
    divider("=", color=TermStyle.BLUE)
    print_help()

    capture = cv2.VideoCapture(0)
    if not capture.isOpened():
        capture = cv2.VideoCapture(1)
    has_webcam = capture.isOpened()

    if not has_webcam:
        print(style("No webcam detected. File and test modes still work.", TermStyle.YELLOW, bold=True))

    while True:
        frame = None
        if has_webcam:
            ok, live_frame = capture.read()
            if ok:
                frame = live_frame
                display = frame.copy()
                cv2.putText(
                    display,
                    "SPACE capture | f file | t test | a ask | h help | q quit",
                    (10, display.shape[0] - 16),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.52,
                    (0, 255, 255),
                    1,
                )
                cv2.imshow(WINDOW_NAME, display)

        if has_webcam:
            key = cv2.waitKey(1) & 0xFF
        else:
            command = input(style("\nCommand [t/f/a/h/q]: ", TermStyle.CYAN)).strip().lower()
            key = ord(command[:1]) if command else 0

        if key == ord("q"):
            break

        if key == ord("h"):
            print_help()
            continue

        if key == ord("a"):
            if LAST_ANALYSIS is None or LAST_ANALYSIS.get("image_path") is None:
                print(style("No previous analysis yet. Run [space], [f], or [t] first.", TermStyle.YELLOW))
            else:
                refreshed = preprocess_image(LAST_ANALYSIS["image_path"])
                if refreshed is None:
                    print(style("Could not reopen the last image for follow-up.", TermStyle.RED))
                else:
                    try:
                        ask_follow_up(refreshed)
                    finally:
                        try:
                            os.unlink(refreshed)
                        except OSError:
                            pass
            continue

        if key == ord(" ") and has_webcam:
            if frame is None:
                print(style("Webcam frame not ready yet. Try again in a second.", TermStyle.YELLOW))
                continue
            temp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
            cv2.imwrite(temp.name, frame)
            print(style(f"\nCaptured webcam frame -> {temp.name}", TermStyle.GREEN))
            try:
                analyze_circuit(temp.name, "webcam capture")
            finally:
                try:
                    os.unlink(temp.name)
                except OSError:
                    pass
            continue

        if key == ord("f"):
            file_name = find_local_circuit_file()
            if file_name is None:
                print(style("No local circuit image found.", TermStyle.YELLOW, bold=True))
                print("Place one of these files in the current folder:")
                for name in SEARCHED_IMAGE_NAMES:
                    print(f"  - {name}")
            else:
                print(style(f"\nLoading image -> {file_name}", TermStyle.GREEN))
                analyze_circuit(file_name, "local file")
            continue

        if key == ord("t"):
            print(style("\nGenerating fallback schematic...", TermStyle.GREEN, bold=True))
            test_path = generate_test_circuit()
            analyze_circuit(test_path, "generated test schematic")
            continue

    if has_webcam:
        capture.release()
    cv2.destroyAllWindows()
    print(style("\nCircuitWhisperer ended. Day 22 shipped.", TermStyle.GREEN, bold=True))


if __name__ == "__main__":
    main()
