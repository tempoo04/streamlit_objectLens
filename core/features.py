import numpy as np
import cv2
import pandas as pd


def compute_features(detections: list) -> tuple[pd.DataFrame, list]:
    """
    For each detection, threshold the mask and compute shape descriptors.

    Returns:
        df      — DataFrame with one row per object
        objects — list of dicts that also carry the binary mask and assigned color
    """
    rows    = []
    objects = []

    for det in detections:
        mask_thresh = det["mask_thresh"]
        binary = (det["mask_raw"] > mask_thresh).astype(np.uint8)
        area = int(binary.sum())
        if area < 10:
            continue

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        cnt = max(contours, key=cv2.contourArea)

        perimeter  = float(cv2.arcLength(cnt, True))
        roundness  = (4 * np.pi * area / perimeter ** 2) if perimeter > 0 else 0.0

        x1, y1, x2, y2 = det["box"]
        w, h = x2 - x1, y2 - y1
        aspect_ratio = float(w / h) if h > 0 else 1.0

        if len(cnt) >= 5:
            axes = cv2.fitEllipse(cnt)[1]
            ma, mi = max(axes), min(axes)
            eccentricity = float(np.sqrt(1 - (mi / ma) ** 2)) if ma > 0 else 0.0
        else:
            eccentricity = 0.0

        row = {
            "id":           det["id"],
            "category":     det["category"],
            "score":        round(det["score"], 3),
            "area_px":      area,
            "roundness":    round(roundness, 3),
            "aspect_ratio": round(aspect_ratio, 3),
            "eccentricity": round(eccentricity, 3),
            "perimeter_px": round(perimeter, 1),
        }
        rows.append(row)
        objects.append({**row, "mask": binary})

    df = pd.DataFrame(rows)
    return df, objects
