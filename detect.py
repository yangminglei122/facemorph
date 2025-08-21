from typing import Optional

import cv2
import numpy as np
import mediapipe as mp

# 修复导入路径问题，确保在PyInstaller打包环境中能正确导入
try:
    # 尝试相对导入（开发环境）
    from utils import DetectionResult
except ImportError:
    # 在PyInstaller打包环境中，尝试从项目根目录导入
    import sys
    import os
    if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
        # 添加项目根目录到sys.path
        root_dir = sys._MEIPASS
        if root_dir not in sys.path:
            sys.path.insert(0, root_dir)
    from utils import DetectionResult


class LandmarkDetector:
    """Wrapper for MediaPipe face mesh and pose for static images."""

    def __init__(self) -> None:
        self._mp_face_mesh = mp.solutions.face_mesh
        self._mp_pose = mp.solutions.pose
        self._face_mesh = self._mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
        )
        self._pose = self._mp_pose.Pose(
            static_image_mode=True,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
        )

    def close(self) -> None:
        self._face_mesh.close()
        self._pose.close()

    def detect(self, image_bgr: np.ndarray) -> DetectionResult:
        h, w = image_bgr.shape[:2]
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        face_points = None
        person_center = None

        # Face landmarks
        fm_res = self._face_mesh.process(image_rgb)
        if fm_res.multi_face_landmarks:
            landmarks = fm_res.multi_face_landmarks[0].landmark
            pts = np.array([[lm.x * w, lm.y * h] for lm in landmarks], dtype=np.float32)
            face_points = pts

        # Pose for person center (use shoulders or hips)
        pose_res = self._pose.process(image_rgb)
        if pose_res.pose_landmarks is not None:
            lms = pose_res.pose_landmarks.landmark
            idx_l = self._mp_pose.PoseLandmark.LEFT_SHOULDER.value
            idx_r = self._mp_pose.PoseLandmark.RIGHT_SHOULDER.value
            idx_lh = self._mp_pose.PoseLandmark.LEFT_HIP.value
            idx_rh = self._mp_pose.PoseLandmark.RIGHT_HIP.value
            cand = []
            for a, b in [(idx_l, idx_r), (idx_lh, idx_rh)]:
                la, lb = lms[a], lms[b]
                if la.visibility > 0.5 and lb.visibility > 0.5:
                    ax, ay = la.x * w, la.y * h
                    bx, by = lb.x * w, lb.y * h
                    cand.append(((ax + bx) / 2.0, (ay + by) / 2.0))
            if cand:
                xy = np.mean(np.array(cand, dtype=np.float32), axis=0)
                person_center = (float(xy[0]), float(xy[1]))
        elif face_points is not None:
            cx, cy = np.mean(face_points, axis=0)
            person_center = (float(cx), float(cy))

        return DetectionResult(face_points=face_points, person_center=person_center, image_size=(h, w))
