"""
Air Drawing System (MediaPipe Tasks API - works with mediapipe 0.10.13+)
========================================================================
SETUP (one-time):
  1. Download the model file:
     curl -o hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
  2. Place hand_landmarker.task in the same folder as this script.
  3. pip install mediapipe opencv-python numpy

GESTURES:
  Pinch (thumb+index close)  -> Draw
  Fist (all fingers closed)  -> Erase
  Open palm held 7s          -> Clear canvas (countdown shown)
  2 fingers up               -> Red
  3 fingers up               -> Green
  4 fingers up               -> Blue
  5 fingers up               -> White

KEYBOARD:
  + / =   Increase brush size
  -       Decrease brush size
  Q       Quit
"""

import cv2
import numpy as np
import time
import os
from collections import deque

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# ── Model path ───────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "hand_landmarker.task")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"\nModel file not found: {MODEL_PATH}\n"
        "Download it with:\n"
        "  curl -o hand_landmarker.task https://storage.googleapis.com/mediapipe-models/"
        "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task\n"
        "and place it next to this script.\n"
    )

# ── MediaPipe detector ───────────────────────────────────────
options = mp_vision.HandLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
    num_hands=1,
    min_hand_detection_confidence=0.6,
    min_hand_presence_confidence=0.6,
    min_tracking_confidence=0.5,
    running_mode=mp_vision.RunningMode.VIDEO,
)
detector = mp_vision.HandLandmarker.create_from_options(options)

# ── Colours (indexed by non-thumb fingers up) ────────────────
COLOURS = {
    2: ((0, 0, 255),     "Red"),
    3: ((0, 255, 0),     "Green"),
    4: ((255, 0, 0),     "Blue"),
    5: ((255, 255, 255), "White"),
}
current_colour      = (0, 0, 255)
current_colour_name = "Red"

# ── State ────────────────────────────────────────────────────
brush_size      = 6
eraser_size     = 40
prev_point      = None
canvas          = None          # initialised on first frame
smooth_buffer   = deque(maxlen=5)
palm_open_start = None
CLEAR_HOLD_TIME = 7.0

# ── Hand skeleton connections ────────────────────────────────
CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),
    (0,17),
]


def get_finger_states(lm, w, h):
    index_up  = lm[8].y  < lm[6].y
    middle_up = lm[12].y < lm[10].y
    ring_up   = lm[16].y < lm[14].y
    pinky_up  = lm[20].y < lm[18].y
    thumb_up  = lm[4].x  < lm[3].x   # mirrored right hand

    non_thumb_up = sum([index_up, middle_up, ring_up, pinky_up])

    tx, ty   = lm[4].x * w, lm[4].y * h
    ix, iy   = lm[8].x * w, lm[8].y * h
    is_pinching   = np.hypot(tx - ix, ty - iy) < 40
    is_fist       = not (index_up or middle_up or ring_up or pinky_up)
    is_open_palm  = index_up and middle_up and ring_up and pinky_up

    return non_thumb_up, is_fist, is_pinching, is_open_palm, (int(ix), int(iy))


def smooth_point(pt):
    smooth_buffer.append(pt)
    return (int(np.mean([p[0] for p in smooth_buffer])),
            int(np.mean([p[1] for p in smooth_buffer])))


def draw_skeleton(frame, lm, w, h):
    pts = [(int(l.x * w), int(l.y * h)) for l in lm]
    for a, b in CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (70, 70, 70), 1)
    for p in pts:
        cv2.circle(frame, p, 3, (110, 110, 110), -1)


def draw_ui(frame, colour, colour_name, brush, erasing, countdown):
    h, w = frame.shape[:2]

    # Top bar
    cv2.rectangle(frame, (0, 0), (w, 55), (20, 20, 20), -1)
    cv2.circle(frame, (30, 27), 16, colour, -1)
    cv2.circle(frame, (30, 27), 16, (200, 200, 200), 1)
    cv2.putText(frame, colour_name,    (55, 33),     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220,220,220), 1)
    cv2.putText(frame, f"Brush: {brush}px", (w//2-50, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180,180,180), 1)
    mode_txt = "ERASING" if erasing else "DRAWING"
    mode_clr = (0, 100, 255) if erasing else (100, 255, 100)
    cv2.putText(frame, mode_txt, (w-170, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.7, mode_clr, 2)

    # Countdown bar + text
    if countdown is not None:
        pct   = min(countdown / CLEAR_HOLD_TIME, 1.0)
        bar_w = int(w * pct)
        secs  = max(1, int(CLEAR_HOLD_TIME - countdown) + 1)
        cv2.rectangle(frame, (0, h-12), (w, h),     (40,40,40),    -1)
        cv2.rectangle(frame, (0, h-12), (bar_w, h), (0, 60, 220),  -1)
        cv2.putText(frame, f"Clearing in {secs}s...",
                    (w//2 - 140, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 80, 255), 3)

    # Legend
    cv2.putText(frame, "Pinch=Draw  Fist=Erase  Palm 7s=Clear",
                (10, h-34), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160,160,160), 1)
    cv2.putText(frame, "2/3/4/5 fingers=R/G/B/W   +/-=Brush   Q=Quit",
                (10, h-18), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160,160,160), 1)


# ── Camera ───────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print(__doc__)
print("Camera starting… press Q to quit.\n")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w  = frame.shape[:2]

    if canvas is None:
        canvas = np.zeros((h, w, 4), dtype=np.uint8)

    # Detect
    rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect_for_video(mp_img, int(time.time() * 1000))

    countdown_val = None
    erasing       = False

    if result.hand_landmarks:
        lm = result.hand_landmarks[0]
        draw_skeleton(frame, lm, w, h)

        non_thumb_up, is_fist, is_pinching, is_open_palm, (ix, iy) = \
            get_finger_states(lm, w, h)
        smooth_pt = smooth_point((ix, iy))

        # Colour switch (fingers up, not drawing/erasing)
        if not is_pinching and not is_fist and non_thumb_up in COLOURS:
            current_colour, current_colour_name = COLOURS[non_thumb_up]
            prev_point = None

        # Clear countdown
        if is_open_palm:
            if palm_open_start is None:
                palm_open_start = time.time()
            elapsed       = time.time() - palm_open_start
            countdown_val = elapsed
            if elapsed >= CLEAR_HOLD_TIME:
                canvas          = np.zeros((h, w, 4), dtype=np.uint8)
                palm_open_start = None
                smooth_buffer.clear()
        else:
            palm_open_start = None

        # Erase
        if is_fist:
            erasing = True
            cv2.circle(canvas, smooth_pt, eraser_size, (0, 0, 0, 0), -1)
            cv2.circle(frame,  smooth_pt, eraser_size, (0, 80, 200), 2)
            prev_point = None

        # Draw
        elif is_pinching and not is_open_palm:
            if prev_point is not None:
                cv2.line(canvas, prev_point, smooth_pt,
                         (*current_colour, 255), brush_size, lineType=cv2.LINE_AA)
            prev_point = smooth_pt
            cv2.circle(frame, smooth_pt, max(brush_size // 2, 3), current_colour, -1)

        else:
            prev_point = None
            cv2.circle(frame, smooth_pt, 5, current_colour, -1)

    else:
        prev_point      = None
        palm_open_start = None
        smooth_buffer.clear()

    # Blend canvas
    alpha      = canvas[:, :, 3:4] / 255.0
    frame      = (frame * (1 - alpha) + canvas[:, :, :3] * alpha).astype(np.uint8)

    draw_ui(frame, current_colour, current_colour_name, brush_size, erasing, countdown_val)

    cv2.imshow("Air Drawing", frame)

    key = cv2.waitKey(1) & 0xFF
    if key in (ord('q'), ord('Q')):
        break
    elif key in (ord('+'), ord('=')):
        brush_size = min(brush_size + 2, 60)
    elif key == ord('-'):
        brush_size = max(brush_size - 2, 2)

cap.release()
detector.close()
cv2.destroyAllWindows()
print("Bye!")