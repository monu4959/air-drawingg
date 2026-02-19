"""
Air Drawing System (MediaPipe Tasks API - works with mediapipe 0.10.13+)
========================================================================
SETUP (one-time):
  1. Download the model file:
     curl -o hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
  2. Place hand_landmarker.task in the same folder as this script.
  3. pip install mediapipe opencv-python numpy

GESTURES:
  Index only (thumb closed)              -> Draw with index tip
  Full fist                              -> Erase (large)
  Only pinky up, hold 3s                 -> Clear canvas

  COLOUR (2s lock after switching):
  Thumb only                             -> Red
  Index + Middle (thumb closed)          -> Green
  Index + Middle + Ring (thumb closed)   -> Blue

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

# ── Colours ───────────────────────────────────────────────────
COLOUR_RED   = ((0, 0, 255),     "Red")
COLOUR_GREEN = ((0, 255, 0),     "Green")
COLOUR_BLUE  = ((255, 0, 0),     "Blue")

current_colour      = COLOUR_RED[0]
current_colour_name = COLOUR_RED[1]

COLOUR_LOCK_DURATION = 2.0   # seconds to wait after colour switch before drawing
colour_changed_at    = None  # timestamp of last colour change

# ── State ─────────────────────────────────────────────────────
brush_size      = 6
eraser_size     = 80          # bigger eraser
prev_point      = None
canvas          = None
smooth_buffer   = deque(maxlen=5)
pinky_start     = None
CLEAR_HOLD_TIME = 3.0

# ── Hand skeleton connections ─────────────────────────────────
CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),
    (0,17),
]


def get_finger_states(lm, w, h):
    thumb_up  = lm[4].x  < lm[3].x   # mirrored right hand
    index_up  = lm[8].y  < lm[6].y
    middle_up = lm[12].y < lm[10].y
    ring_up   = lm[16].y < lm[14].y
    pinky_up  = lm[20].y < lm[18].y
    ix = int(lm[8].x * w)
    iy = int(lm[8].y * h)
    return thumb_up, index_up, middle_up, ring_up, pinky_up, (ix, iy)


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


def draw_ui(frame, colour, colour_name, brush, mode, clear_elapsed, colour_lock_remaining):
    h, w = frame.shape[:2]

    # Top bar background
    cv2.rectangle(frame, (0, 0), (w, 55), (20, 20, 20), -1)

    # Colour swatch
    cv2.circle(frame, (30, 27), 16, colour, -1)
    cv2.circle(frame, (30, 27), 16, (200, 200, 200), 1)
    cv2.putText(frame, colour_name, (55, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1)

    # Brush size
    cv2.putText(frame, f"Brush: {brush}px", (w//2 - 50, 33),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)

    # Mode label
    mode_map = {
        "DRAWING":  (100, 255, 100),
        "ERASING":  (0, 100, 255),
        "CLEARING": (0, 80, 255),
        "LOCKED":   (0, 200, 255),
        "IDLE":     (160, 160, 160),
    }
    cv2.putText(frame, mode, (w - 175, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                mode_map.get(mode, (160, 160, 160)), 2)

    # Colour lock countdown bar (below top bar)
    if colour_lock_remaining is not None and colour_lock_remaining > 0:
        pct   = colour_lock_remaining / COLOUR_LOCK_DURATION
        bar_w = int(w * pct)
        cv2.rectangle(frame, (0, 55), (w, 68),     (40, 40, 40),  -1)
        cv2.rectangle(frame, (0, 55), (bar_w, 68), colour,        -1)
        cv2.putText(frame, f"Ready in {colour_lock_remaining:.1f}s",
                    (w//2 - 70, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1)

    # Clear progress bar at bottom
    if clear_elapsed is not None:
        pct   = min(clear_elapsed / CLEAR_HOLD_TIME, 1.0)
        bar_w = int(w * pct)
        cv2.rectangle(frame, (0, h - 14), (w, h),     (40, 40, 40),  -1)
        cv2.rectangle(frame, (0, h - 14), (bar_w, h), (0, 60, 220),  -1)
        cv2.putText(frame, "Clearing...", (10, h - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    # Legend
    y_off = h - 34 if clear_elapsed is None else h - 20
    cv2.putText(frame, "Index=Draw  Fist=Erase  Pinky 3s=Clear",
                (10, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 160, 160), 1)
    cv2.putText(frame, "Thumb=Red  I+M=Green  I+M+R=Blue   +/-=Brush   Q=Quit",
                (10, y_off + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 160, 160), 1)


# ── Camera ────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print(__doc__)
print("Camera starting... press Q to quit.\n")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w  = frame.shape[:2]

    if canvas is None:
        canvas = np.zeros((h, w, 4), dtype=np.uint8)

    # ── Detect ────────────────────────────────────────────────
    rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect_for_video(mp_img, int(time.time() * 1000))

    clear_elapsed        = None
    mode                 = "IDLE"
    colour_lock_remaining = None

    # How long since colour changed
    if colour_changed_at is not None:
        elapsed_since_change  = time.time() - colour_changed_at
        colour_lock_remaining = max(0.0, COLOUR_LOCK_DURATION - elapsed_since_change)
        if colour_lock_remaining == 0.0:
            colour_changed_at = None   # lock expired

    if result.hand_landmarks:
        lm = result.hand_landmarks[0]
        draw_skeleton(frame, lm, w, h)

        thumb_up, index_up, middle_up, ring_up, pinky_up, (ix, iy) = \
            get_finger_states(lm, w, h)
        smooth_pt = smooth_point((ix, iy))

        # ── Derived gesture booleans ───────────────────────────
        only_thumb  = thumb_up  and not index_up and not middle_up and not ring_up and not pinky_up
        only_index  = index_up  and not thumb_up and not middle_up and not ring_up and not pinky_up
        only_pinky  = pinky_up  and not thumb_up and not index_up  and not middle_up and not ring_up
        idx_mid     = not thumb_up and index_up and middle_up and not ring_up and not pinky_up
        idx_mid_rng = not thumb_up and index_up and middle_up and ring_up and not pinky_up
        is_fist     = not thumb_up and not index_up and not middle_up and not ring_up and not pinky_up

        # ── COLOUR SELECTION ──────────────────────────────────
        new_colour = None
        if only_thumb:
            new_colour = COLOUR_RED
        elif idx_mid:
            new_colour = COLOUR_GREEN
        elif idx_mid_rng:
            new_colour = COLOUR_BLUE

        if new_colour is not None:
            col, name = new_colour
            if name != current_colour_name:          # only reset lock on actual change
                current_colour      = col
                current_colour_name = name
                colour_changed_at   = time.time()
                colour_lock_remaining = COLOUR_LOCK_DURATION
            prev_point  = None
            pinky_start = None
            mode        = "LOCKED" if colour_lock_remaining and colour_lock_remaining > 0 else "IDLE"

        # ── CLEAR: only pinky, hold 3s ─────────────────────────
        elif only_pinky:
            if pinky_start is None:
                pinky_start = time.time()
            clear_elapsed = time.time() - pinky_start
            mode          = "CLEARING"
            if clear_elapsed >= CLEAR_HOLD_TIME:
                canvas        = np.zeros((h, w, 4), dtype=np.uint8)
                pinky_start   = None
                smooth_buffer.clear()
            prev_point = None

        # ── ERASE: fist ────────────────────────────────────────
        elif is_fist:
            pinky_start = None
            mode        = "ERASING"
            cv2.circle(canvas, smooth_pt, eraser_size, (0, 0, 0, 0), -1)
            cv2.circle(frame,  smooth_pt, eraser_size, (0, 80, 200), 2)
            prev_point = None

        # ── DRAW: only index, no colour lock active ────────────
        elif only_index:
            pinky_start = None
            if colour_lock_remaining and colour_lock_remaining > 0:
                # Still locked — show mode but don't draw
                mode       = "LOCKED"
                prev_point = None
            else:
                mode = "DRAWING"
                if prev_point is not None:
                    cv2.line(canvas, prev_point, smooth_pt,
                             (*current_colour, 255), brush_size, lineType=cv2.LINE_AA)
                prev_point = smooth_pt
                cv2.circle(frame, smooth_pt, max(brush_size // 2, 3), current_colour, -1)

        else:
            pinky_start = None
            prev_point  = None
            mode        = "IDLE"

        # Cursor dot when not actively drawing
        if mode != "DRAWING":
            cv2.circle(frame, smooth_pt, 5, current_colour, -1)

    else:
        prev_point  = None
        pinky_start = None
        smooth_buffer.clear()
        mode = "IDLE"

    # ── Blend canvas ──────────────────────────────────────────
    alpha = canvas[:, :, 3:4] / 255.0
    frame = (frame * (1 - alpha) + canvas[:, :, :3] * alpha).astype(np.uint8)

    draw_ui(frame, current_colour, current_colour_name, brush_size,
            mode, clear_elapsed, colour_lock_remaining)

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