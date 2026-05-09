import cv2
import mediapipe as mp
import time
import sys

# ── Console Styling ─────────────────────────────────────

CYAN="\033[96m"; GREEN="\033[92m"; YELLOW="\033[93m"
RED="\033[91m"; BOLD="\033[1m"; DIM="\033[2m"; RESET="\033[0m"

def clear(): print("\033[2J\033[H", end="")

def banner():
    print(f"{CYAN}{BOLD}")
    print("  ██████╗ ██╗     ██╗███╗   ██╗██╗  ██╗")
    print("  ██╔══██╗██║     ██║████╗  ██║██║ ██╔╝")
    print("  ██████╔╝██║     ██║██╔██╗ ██║█████╔╝ ")
    print("  ██╔══██╗██║     ██║██║╚██╗██║██╔═██╗ ")
    print("  ██████╔╝███████╗██║██║ ╚████║██║  ██╗")
    print("  ╚═════╝ ╚══════╝╚═╝╚═╝  ╚═══╝╚═╝  ╚═╝")
    print(f"{DIM}BlinkLock v1.3 — Vision Lock System{RESET}\n")

def instructions():
    print(f"{BOLD}── HOW TO USE ─────────────────────────────{RESET}")
    print(f"{GREEN}👁 Blink 3 times fast{RESET} → LOCK")
    print(f"{YELLOW}😉 Slow wink (one eye){RESET} → UNLOCK")
    print(f"{RED}Q{RESET} → Quit")
    print(f"{BOLD}──────────────────────────────────────────{RESET}\n")

# ── Setup ─────────────────────────────────────────────

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Camera not found"); sys.exit(1)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# ── Landmarks ─────────────────────────────────────────

LEFT_TOP=[159,160,161]; LEFT_BOTTOM=[145,144,153]; LEFT_L=33; LEFT_R=133
RIGHT_TOP=[386,387,388]; RIGHT_BOTTOM=[374,373,380]; RIGHT_L=362; RIGHT_R=263

def ear(lm, top, bottom, l, r):
    v = sum(abs(lm[t].y-lm[b].y) for t,b in zip(top,bottom))/len(top)
    h = abs(lm[l].x-lm[r].x)
    return v/h if h else 0

# ── Parameters ─────────────────────────────────────────

EAR_THRESH = 0.24
BLINK_MAX = 0.25
WINK_MIN = 0.35
BLINK_WINDOW = 1.0
BLINK_TARGET = 3

# ── States ─────────────────────────────────────────────

IDLE="IDLE"; COUNT="COUNT"; LOCKED="LOCKED"

state = IDLE
blink_count = 0
start_time = 0

both_closed = False
close_start = 0

left_only = False
right_only = False
wink_start = 0

# ── Start UI ──────────────────────────────────────────

clear()
banner()
instructions()
input(f"{YELLOW}Press ENTER to start...{RESET}")

# ── Loop ──────────────────────────────────────────────

while True:
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.flip(frame,1)
    h,w = frame.shape[:2]

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = face_mesh.process(rgb)

    if res.multi_face_landmarks:
        lm = res.multi_face_landmarks[0].landmark

        L = ear(lm, LEFT_TOP, LEFT_BOTTOM, LEFT_L, LEFT_R)
        R = ear(lm, RIGHT_TOP, RIGHT_BOTTOM, RIGHT_L, RIGHT_R)

        now = time.time()
        lc = L < EAR_THRESH
        rc = R < EAR_THRESH

        # BLINK
        if lc and rc:
            if not both_closed:
                both_closed = True
                close_start = now
        else:
            if both_closed:
                dur = now - close_start
                both_closed = False
                if dur < BLINK_MAX:
                    if state == IDLE:
                        state = COUNT
                        blink_count = 1
                        start_time = now
                    elif state == COUNT:
                        blink_count += 1

        # WINK
        if lc and not rc:
            if not left_only:
                left_only = True
                wink_start = now
        else:
            if left_only:
                dur = now - wink_start
                left_only = False
                if dur > WINK_MIN and state == LOCKED:
                    state = IDLE
                    blink_count = 0
                    print(f"{GREEN}Unlocked (left wink){RESET}")

        if rc and not lc:
            if not right_only:
                right_only = True
                wink_start = now
        else:
            if right_only:
                dur = now - wink_start
                right_only = False
                if dur > WINK_MIN and state == LOCKED:
                    state = IDLE
                    blink_count = 0
                    print(f"{GREEN}Unlocked (right wink){RESET}")

        # STATE
        if state == COUNT:
            if now - start_time > BLINK_WINDOW:
                state = IDLE
                blink_count = 0
            elif blink_count >= BLINK_TARGET:
                state = LOCKED
                print(f"{RED}LOCKED{RESET}")

        # HUD
        cv2.rectangle(frame,(10,10),(260,120),(20,20,20),-1)

        cv2.putText(frame,f"L:{L:.2f} R:{R:.2f}",(20,40),
                    cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,255),2)

        color = (0,255,0) if state==IDLE else (0,200,255) if state==COUNT else (0,0,255)
        cv2.putText(frame,f"{state}",(20,75),
                    cv2.FONT_HERSHEY_SIMPLEX,0.8,color,2)

        cv2.putText(frame,f"Blinks:{blink_count}",(20,105),
                    cv2.FONT_HERSHEY_SIMPLEX,0.6,(200,200,200),2)

        if state == LOCKED:
            overlay = frame.copy()
            cv2.rectangle(overlay,(0,0),(w,h),(0,0,0),-1)
            frame = cv2.addWeighted(overlay,0.85,frame,0.15,0)

            cv2.putText(frame,"LOCKED",(w//2-120,h//2),
                        cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),4)
            cv2.putText(frame,"Wink to unlock;)",(w//2-130,h//2+50),
                        cv2.FONT_HERSHEY_SIMPLEX,0.7,(200,200,200),2)

    cv2.imshow("BlinkLock v1.3",frame)

    if cv2.waitKey(1)&0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print(f"\n{RED}BlinkLock stopped.{RESET}")