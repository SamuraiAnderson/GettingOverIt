"""Single-frame gold-pot centre detector matching the TAS colour contour definition.

The TAS video path tracks over time and then fuses box vs colour. A still game
frame has no tracker history, so this keeps the colour-contour centroid and the
same HSV / area / aspect gates, scaled from the original 640-wide working image.
"""
from __future__ import annotations

import cv2
import numpy as np

TAS_WORKING_WIDTH = 640
DETECTOR_VERSION = "tas_color_contour_v1"
LIVE_DETECTOR_VERSION = "live_dark_contour_v1"


def detect_pot_centre(
    bgr: np.ndarray,
    search_center_px=None,
    search_radius_px: float | None = None,
) -> dict:
    image = np.asarray(bgr)
    if image.ndim != 3:
        raise ValueError("detector expects an HxWx3 BGR image")
    height, width = image.shape[:2]
    scale = width / TAS_WORKING_WIDTH
    small = cv2.resize(image, (TAS_WORKING_WIDTH, max(1, int(round(height / scale)))))
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    center_small = None
    radius_small = None
    if search_center_px is not None and search_radius_px:
        center_small = np.asarray(search_center_px, dtype=np.float64) / scale
        radius_small = float(search_radius_px) / scale
    hues = [12, 24, 28, 36, 48, 60, 72]
    candidates = []
    for hue in hues:
        for saturation in (40, 80, 110, 155):
            mask = cv2.inRange(
                hsv,
                np.array([max(0, hue - 9), saturation, 15]),
                np.array([min(179, hue + 9), 255, 255]),
            )
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                area = cv2.contourArea(contour)
                if not (350 < area < 2300 and 20 < w < 60 and 23 < h < 65 and 0.55 < w / h < 1.55):
                    continue
                if area / (w * h) < 0.5:
                    continue
                moments = cv2.moments(contour)
                if moments["m00"] <= 0:
                    continue
                pos = np.array([moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]])
                if center_small is not None and np.linalg.norm(pos - center_small) > radius_small:
                    continue
                confidence = float(np.exp(-0.5 * (np.log(area / 1100.0) / 0.6) ** 2))
                candidates.append((confidence, pos, area, hue))
    if not candidates:
        return {
            "detected_px": None,
            "confidence": 0.0,
            "detector_version": DETECTOR_VERSION,
            "reason": "no_gold_contour_in_window" if center_small is not None else "no_gold_contour",
        }
    confidence, pos, area, hue = max(candidates, key=lambda item: item[0])
    if confidence < 0.45:
        return {
            "detected_px": None,
            "confidence": confidence,
            "detector_version": DETECTOR_VERSION,
            "reason": "low_confidence",
        }
    return {
        "detected_px": [float(pos[0] * scale), float(pos[1] * scale)],
        "confidence": confidence,
        "area_small": float(area),
        "hue": int(hue),
        "detector_version": DETECTOR_VERSION,
        "reason": "ok",
    }


def detect_live_pot_centre(
    bgr: np.ndarray,
    search_center_px,
    search_radius_px: float,
) -> dict:
    """Visible pot centre on the live dark-brown pot.

    TAS gold HSV shares hue with the spawn sandstone, so sitting frames merge
    pot and rock. The live pot is dark (V≈38); the rock is the same hue but
    much brighter. This detector uses a low-value mask inside a body window.
    """
    image = np.asarray(bgr)
    if image.ndim != 3:
        raise ValueError("detector expects an HxWx3 BGR image")
    if search_center_px is None or not search_radius_px:
        return {
            "detected_px": None,
            "confidence": 0.0,
            "detector_version": LIVE_DETECTOR_VERSION,
            "reason": "search_window_required",
        }
    height, width = image.shape[:2]
    scale = width / TAS_WORKING_WIDTH
    small = cv2.resize(image, (TAS_WORKING_WIDTH, max(1, int(round(height / scale)))))
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    center_small = np.asarray(search_center_px, dtype=np.float64) / scale
    radius_small = float(search_radius_px) / scale
    candidates = []
    kernel = np.ones((3, 3), np.uint8)
    for vmax in (42, 52, 62):
        mask = cv2.inRange(
            hsv,
            np.array([8, 20, 4]),
            np.array([55, 255, vmax]),
        )
        yy, xx = np.ogrid[:small.shape[0], :small.shape[1]]
        window = (xx - center_small[0]) ** 2 + (yy - center_small[1]) ** 2 <= radius_small ** 2
        mask = np.where(window, mask, 0).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            if not (280 < area < 2600 and 16 < w < 70 and 18 < h < 75 and 0.50 < w / h < 1.70):
                continue
            if area / (w * h) < 0.45:
                continue
            moments = cv2.moments(contour)
            if moments["m00"] <= 0:
                continue
            pos = np.array([moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]])
            if np.linalg.norm(pos - center_small) > radius_small:
                continue
            peri = cv2.arcLength(contour, True)
            circular = 0.0 if peri <= 0 else float(4.0 * np.pi * area / (peri * peri))
            area_score = float(np.exp(-0.5 * (np.log(area / 1100.0) / 0.6) ** 2))
            confidence = area_score * (0.45 + 0.55 * min(circular, 1.0))
            candidates.append((confidence, pos, area, vmax, circular))
    if not candidates:
        return {
            "detected_px": None,
            "confidence": 0.0,
            "detector_version": LIVE_DETECTOR_VERSION,
            "reason": "no_dark_contour_in_window",
        }
    confidence, pos, area, vmax, circular = max(candidates, key=lambda item: item[0])
    if confidence < 0.35:
        return {
            "detected_px": None,
            "confidence": confidence,
            "detector_version": LIVE_DETECTOR_VERSION,
            "reason": "low_confidence",
        }
    return {
        "detected_px": [float(pos[0] * scale), float(pos[1] * scale)],
        "confidence": float(confidence),
        "area_small": float(area),
        "vmax": int(vmax),
        "circularity": float(circular),
        "detector_version": LIVE_DETECTOR_VERSION,
        "reason": "ok",
    }
