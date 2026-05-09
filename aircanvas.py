import cv2
import mediapipe as mp
import numpy as np
import math

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("ERROR: No webcam found.")
    exit(1)

ret, test_frame = cap.read()
FRAME_H, FRAME_W = test_frame.shape[:2]

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.6
)
mp_draw = mp.solutions.drawing_utils

canvas = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)

THUMB_TIP = 4
INDEX_TIP = 8

prev_x, prev_y = None, None
was_drawing = False

PINCH_THRESHOLD = 25   # distance in pixels to trigger drawing
PEN_SIZE = 6           # thickness of the pen stroke

COLORS = [
    (0, 255, 0),     # Green
    (0, 0, 255),     # Red
    (255, 0, 127),   # Violet
    (225, 0, 0),     # Blue
    (0, 225, 225),   # Yellow
    (225, 225, 225), # White
]

COLOR_NAMES = ["Green", "Red", "Violet", "Blue", "Yellow", "White"]
current_color_index = 0

def get_distance(lm1, lm2, w, h):
    x1, y1 = int(lm1.x * w), int(lm1.y * h)
    x2, y2 = int(lm2.x * w), int(lm2.y * h)
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

#instructions:
print("\n✨ AirCanvas by Otabek('s AI😅) ✨")
print("Gestures:")
print(f"  • Pinch (Index+Thumb) → Draw (threshold {PINCH_THRESHOLD}px)")
print("Keys:")
print("  • Number keys (1,2,3...) → Switch colors")
print("  • 'c' → Clear canvas")
print("  • 'q' → Quit\n")

# Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    drawing_now = False

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        landmarks = hand_landmarks.landmark
        thumb = landmarks[THUMB_TIP]
        index = landmarks[INDEX_TIP]

        distance = get_distance(thumb, index, FRAME_W, FRAME_H)
        ix, iy = int(index.x * FRAME_W), int(index.y * FRAME_H)

        cv2.circle(frame, (ix, iy), 6, COLORS[current_color_index], -1)

        if distance < PINCH_THRESHOLD:
            drawing_now = True
            if was_drawing and prev_x is not None:
                cv2.line(canvas, (prev_x, prev_y), (ix, iy),
                         COLORS[current_color_index], thickness=PEN_SIZE)
            prev_x, prev_y = ix, iy
        else:
            prev_x, prev_y = None, None

        cv2.putText(frame, f"Distance: {distance:.0f}px", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Threshold: {PINCH_THRESHOLD}px", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    else:
        prev_x, prev_y = None, None

    was_drawing = drawing_now

    status = "🎨 DRAWING" if drawing_now else "…waiting"
    color = COLORS[current_color_index] if drawing_now else (150, 150, 150)
    cv2.putText(frame, status, (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)

    cv2.putText(frame, f"Color: {COLOR_NAMES[current_color_index]}",
                (10, FRAME_H - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                COLORS[current_color_index], 2)

    mask = canvas > 0
    frame[mask] = canvas[mask]

    cv2.putText(frame, "Controls: [1-6] Colors | 'c' Clear | 'q' Quit",
                (10, FRAME_H - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (200, 200, 200), 2)

    cv2.imshow("AirCanvas - Gwordal", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        canvas = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
        print("🧽 Canvas cleared!")
    elif ord('1') <= key <= ord('9'):
        idx = key - ord('1')
        if idx < len(COLORS):
            current_color_index = idx
            print(f"🎨 Color switched to {COLOR_NAMES[current_color_index]}")

cap.release()
cv2.destroyAllWindows()
print("\nAirCanvas ended. Keep creating, Otabek!")
