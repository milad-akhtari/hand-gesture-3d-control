import sys
import os
import math
import threading
import time
from dataclasses import dataclass
from typing import Optional, List

import cv2
import mediapipe as mp

from direct.showbase.ShowBase import ShowBase
from panda3d.core import (
    AmbientLight, DirectionalLight, WindowProperties,
    Point3, Filename
)

# ============================
# Settings
# ============================
SHOW_DEBUG = False

# Start background: "black" or "white"
START_BG = "black"   # change to "white" if you prefer

# Rotation feel
ROTATE_RANGE_H = 260.0
ROTATE_RANGE_P = 180.0
ROTATE_LERP    = 0.35

# Roll (hand twist)
ROLL_RANGE = 180.0
ROLL_LERP  = 0.25

# Zoom
ZOOM_SENS = 55.0

# ============================
# Inertia settings
# ============================
INERTIA_ENABLED = True
VEL_LERP = 0.35
FRICTION_PER_SEC = 0.85
VEL_CUTOFF = 2.0


# ----------------------------
# File picker (Windows)
# ----------------------------

def pick_model_file() -> Optional[str]:
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)

        filetypes = [
            ("3D Models", "*.glb *.gltf *.bam *.egg"),
            ("glTF Binary (*.glb)", "*.glb"),
            ("glTF (*.gltf)", "*.gltf"),
            ("Panda3D Models (*.bam;*.egg)", "*.bam *.egg"),
            ("All files", "*.*"),
        ]
        path = filedialog.askopenfilename(
            title="Select a 3D model file",
            filetypes=filetypes
        )
        root.destroy()
        return path if path else None
    except Exception as e:
        print("File picker failed:", e)
        return None


# ----------------------------
# Hand tracking (threaded)
# ----------------------------

@dataclass
class HandState:
    hands: List[dict]
    debug_frame: Optional[any] = None


class HandTracker(threading.Thread):
    def __init__(self, camera_index: int = 0, show_debug: bool = True):
        super().__init__(daemon=True)
        self.camera_index = camera_index
        self.show_debug = show_debug
        self._stop = False
        self.latest: HandState = HandState(hands=[], debug_frame=None)

        self.mp_hands = mp.solutions.hands

    def stop(self):
        self._stop = True

    @staticmethod
    def _dist(a, b) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

    @staticmethod
    def _finger_extended(lm, tip, pip) -> bool:
        return lm[tip].y < lm[pip].y

    def _is_open_palm(self, lm) -> bool:
        index_up  = self._finger_extended(lm, 8, 6)
        middle_up = self._finger_extended(lm, 12, 10)
        ring_up   = self._finger_extended(lm, 16, 14)
        pinky_up  = self._finger_extended(lm, 20, 18)
        extended_count = sum([index_up, middle_up, ring_up, pinky_up])

        wrist = (lm[0].x, lm[0].y)
        mid_tip = (lm[12].x, lm[12].y)
        palm_scale = self._dist(wrist, (lm[9].x, lm[9].y)) + 1e-6
        openness = self._dist(mid_tip, wrist) / palm_scale

        return (extended_count >= 3) and (openness > 1.6)

    @staticmethod
    def _hand_twist_angle_deg(lm) -> float:
        x1, y1 = lm[5].x, lm[5].y
        x2, y2 = lm[17].x, lm[17].y
        return math.degrees(math.atan2((y2 - y1), (x2 - x1)))

    def run(self):
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            print("ERROR: Could not open camera.")
            return

        with self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        ) as hands:

            while not self._stop:
                ok, frame = cap.read()
                if not ok:
                    time.sleep(0.01)
                    continue

                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = hands.process(rgb)

                out_hands = []
                if res.multi_hand_landmarks:
                    for hand_lms in res.multi_hand_landmarks:
                        lm = hand_lms.landmark

                        wrist = (lm[0].x, lm[0].y)
                        index_mcp = (lm[5].x, lm[5].y)
                        middle_mcp = (lm[9].x, lm[9].y)

                        palm_point = ((wrist[0] + index_mcp[0] + middle_mcp[0]) / 3.0,
                                      (wrist[1] + index_mcp[1] + middle_mcp[1]) / 3.0)

                        out_hands.append({
                            "open_palm": self._is_open_palm(lm),
                            "palm_point": palm_point,
                            "twist_deg": self._hand_twist_angle_deg(lm)
                        })

                self.latest = HandState(hands=out_hands, debug_frame=None)

        cap.release()


# ----------------------------
# Panda3D App
# ----------------------------

class Gesture3DApp(ShowBase):
    def __init__(self, model_path: Optional[str] = None, show_debug: bool = True):
        super().__init__()
        self.disableMouse()

        props = WindowProperties()
        props.setTitle("Hand 3D Viewer (Rotate+Inertia, 2-hands Zoom, B=Background, ESC=Quit)")
        self.win.requestProperties(props)

        # Background toggle state
        self.bg_mode = START_BG.lower().strip()
        self._apply_background()

        # Lighting
        ambient = AmbientLight("ambient")
        ambient.setColor((0.45, 0.45, 0.45, 1))
        self.render.setLight(self.render.attachNewNode(ambient))

        dlight = DirectionalLight("dlight")
        dlight.setColor((0.9, 0.9, 0.9, 1))
        dlnp = self.render.attachNewNode(dlight)
        dlnp.setHpr(45, -45, 0)
        self.render.setLight(dlnp)

        self.root_np = self.render.attachNewNode("model_root")

        self.model = self._load_model(model_path)
        self.model.reparentTo(self.root_np)
        self._auto_fit_model()

        self.tracker = HandTracker(camera_index=0, show_debug=show_debug)
        self.tracker.start()

        # Rotate state
        self.rotate_active = False
        self.grab_start_pt = None
        self.grab_start_h = 0.0
        self.grab_start_p = 0.0
        self.grab_start_twist = 0.0
        self.grab_start_r = 0.0
        self.roll_smoothed = 0.0

        # Zoom state
        self.zoom_active = False
        self.prev_twohand_dist = None

        # Inertia state
        self.prev_hpr = self.root_np.getHpr()
        self.vel_h = 0.0
        self.vel_p = 0.0
        self.vel_r = 0.0
        self.last_time = time.time()

        self.taskMgr.add(self.update_gestures, "update_gestures")
        self.accept("escape", self.clean_exit)
        self.accept("b", self.toggle_background)
        self.accept("B", self.toggle_background)

    def _apply_background(self):
        if self.bg_mode == "white":
            # White background
            self.setBackgroundColor(1, 1, 1, 1)
        else:
            # Black background
            self.setBackgroundColor(0, 0, 0, 1)

    def toggle_background(self):
        self.bg_mode = "white" if self.bg_mode != "white" else "black"
        self._apply_background()
        print(f"Background: {self.bg_mode}")

    def _load_model(self, model_path: Optional[str]):
        if model_path:
            try:
                fn = Filename.fromOsSpecific(model_path)
                fn.makeAbsolute()
                print("Loading model:", fn.toOsSpecific())

                m = self.loader.loadModel(fn)
                if m:
                    print("✅ Model loaded OK.")
                    return m
                else:
                    print("❌ Failed to load model (returned None).")
            except Exception as e:
                print(f"❌ Could not load model '{model_path}': {e}")

        print("⚠️ Using default panda model.")
        m = self.loader.loadModel("models/panda")
        m.setScale(0.35)
        m.setHpr(180, 0, 0)
        return m

    def _auto_fit_model(self):
        min_pt = Point3()
        max_pt = Point3()
        ok = self.model.calcTightBounds(min_pt, max_pt)

        if not ok:
            self.cam_distance = 15.0
            self.min_cam = 3.0
            self.max_cam = 80.0
            self._apply_camera()
            return

        center = (min_pt + max_pt) * 0.5
        size = max_pt - min_pt
        radius = max(size.x, size.y, size.z) * 0.5
        radius = max(radius, 0.01)

        self.model.setPos(-center)

        target_radius = 2.5
        s = target_radius / radius
        self.model.setScale(s)

        self.cam_distance = 10.0
        self.min_cam = 3.0
        self.max_cam = 80.0
        self._apply_camera()

    def _apply_camera(self):
        self.camera.setPos(0, -self.cam_distance, 1.8)
        self.camera.lookAt(0, 0, 0)

    def clean_exit(self):
        try:
            self.tracker.stop()
        except Exception:
            pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        self.userExit()

    @staticmethod
    def _clamp(x, a, b):
        return max(a, min(b, x))

    @staticmethod
    def _wrap_angle_deg(a):
        while a > 180:
            a -= 360
        while a < -180:
            a += 360
        return a

    def _update_inertia_velocity_from_current_hpr(self, dt: float):
        if dt <= 1e-6:
            return
        cur = self.root_np.getHpr()
        prev = self.prev_hpr

        dh = self._wrap_angle_deg(cur.x - prev.x)
        dp = self._wrap_angle_deg(cur.y - prev.y)
        dr = self._wrap_angle_deg(cur.z - prev.z)

        vh = dh / dt
        vp = dp / dt
        vr = dr / dt

        self.vel_h = (1 - VEL_LERP) * self.vel_h + VEL_LERP * vh
        self.vel_p = (1 - VEL_LERP) * self.vel_p + VEL_LERP * vp
        self.vel_r = (1 - VEL_LERP) * self.vel_r + VEL_LERP * vr

        self.prev_hpr = cur

    def _apply_inertia(self, dt: float):
        if not INERTIA_ENABLED:
            return

        friction = FRICTION_PER_SEC ** dt
        self.vel_h *= friction
        self.vel_p *= friction
        self.vel_r *= friction

        if abs(self.vel_h) < VEL_CUTOFF: self.vel_h = 0.0
        if abs(self.vel_p) < VEL_CUTOFF: self.vel_p = 0.0
        if abs(self.vel_r) < VEL_CUTOFF: self.vel_r = 0.0

        if self.vel_h == 0 and self.vel_p == 0 and self.vel_r == 0:
            return

        h = self.root_np.getH() + self.vel_h * dt
        p = self.root_np.getP() + self.vel_p * dt
        r = self.root_np.getR() + self.vel_r * dt

        p = self._clamp(p, -80, 80)
        self.root_np.setHpr(h, p, r)
        self.prev_hpr = self.root_np.getHpr()

    def update_gestures(self, task):
        now = time.time()
        dt = now - self.last_time
        self.last_time = now
        dt = max(0.0, min(dt, 0.1))

        state = self.tracker.latest
        open_hands = [h for h in state.hands if h["open_palm"]]

        rotating_now = False

        # ROTATE (1 open hand)
        if len(open_hands) == 1:
            rotating_now = True
            hdata = open_hands[0]
            pt = hdata["palm_point"]
            twist = hdata["twist_deg"]

            if not self.rotate_active:
                self.rotate_active = True
                self.grab_start_pt = pt
                self.grab_start_h = self.root_np.getH()
                self.grab_start_p = self.root_np.getP()
                self.grab_start_twist = twist
                self.grab_start_r = self.root_np.getR()
                self.roll_smoothed = self.root_np.getR()
                self.prev_hpr = self.root_np.getHpr()
            else:
                dx = pt[0] - self.grab_start_pt[0]
                dy = pt[1] - self.grab_start_pt[1]

                target_h = self.grab_start_h + dx * ROTATE_RANGE_H
                target_p = self.grab_start_p - dy * ROTATE_RANGE_P

                d_twist = self._wrap_angle_deg(twist - self.grab_start_twist)
                target_r = self.grab_start_r + self._clamp(d_twist, -ROLL_RANGE, ROLL_RANGE)

                new_h = (1 - ROTATE_LERP) * self.root_np.getH() + ROTATE_LERP * target_h
                new_p = (1 - ROTATE_LERP) * self.root_np.getP() + ROTATE_LERP * target_p
                self.roll_smoothed = (1 - ROLL_LERP) * self.roll_smoothed + ROLL_LERP * target_r

                new_p = self._clamp(new_p, -80, 80)
                self.root_np.setHpr(new_h, new_p, self.roll_smoothed)

                self._update_inertia_velocity_from_current_hpr(dt)

        else:
            self.rotate_active = False
            self.grab_start_pt = None

        if not rotating_now:
            self._apply_inertia(dt)

        # ZOOM (2 open hands)
        if len(open_hands) >= 2:
            p1 = open_hands[0]["palm_point"]
            p2 = open_hands[1]["palm_point"]
            d = math.hypot(p1[0] - p2[0], p1[1] - p2[1])

            if not self.zoom_active:
                self.zoom_active = True
                self.prev_twohand_dist = d
            else:
                delta = d - (self.prev_twohand_dist or d)
                self.prev_twohand_dist = d

                self.cam_distance -= delta * ZOOM_SENS
                self.cam_distance = self._clamp(self.cam_distance, self.min_cam, self.max_cam)
                self._apply_camera()
        else:
            self.zoom_active = False
            self.prev_twohand_dist = None

        return task.cont


def main():
    if len(sys.argv) >= 2:
        model_path = sys.argv[1]
    else:
        model_path = pick_model_file()

    if not model_path:
        print("No model selected. Exiting.")
        return

    model_path = os.path.abspath(model_path)

    app = Gesture3DApp(model_path=model_path, show_debug=SHOW_DEBUG)
    app.run()


if __name__ == "__main__":
    main()
