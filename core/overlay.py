import numpy as np
import cv2

# ── instance color palette ────────────────────────────────────────────────────
PALETTE = [
    (55,  138, 221),   # blue
    (29,  158, 117),   # teal
    (216,  90,  48),   # coral
    (186, 117,  23),   # amber
    (83,   74, 183),   # purple
    (212,  83, 126),   # pink
    (99,  153,  34),   # green
    (136, 135, 128),   # gray
]


def get_instance_color(idx: int) -> tuple:
    return PALETTE[idx % len(PALETTE)]


def _metric_color(t: float) -> np.ndarray:
    """Blue → teal → coral gradient mapped to t ∈ [0, 1]."""
    if t < 0.5:
        s = t * 2
        return np.array([55 + s * (29 - 55), 138 + s * (158 - 138), 221 + s * (117 - 221)])
    else:
        s = (t - 0.5) * 2
        return np.array([29 + s * (216 - 29), 158 + s * (90 - 158), 117 + s * (48 - 117)])


def render_overlay(img_np: np.ndarray, objects: list, color_by: str) -> np.ndarray:
    """
    Composite per-instance masks onto the original image.

    color_by: "instance" → unique palette color per object
              any other string → metric gradient (blue=low, coral=high)
    """
    overlay = img_np.copy().astype(np.float32)

    metric_min, metric_max = 0.0, 1.0
    if color_by != "instance":
        vals = [o[color_by] for o in objects if color_by in o]
        if vals:
            metric_min, metric_max = min(vals), max(vals)

    for i, obj in enumerate(objects):
        mask = obj["mask"]

        if color_by == "instance":
            color = np.array(get_instance_color(i), dtype=np.float32)
        else:
            val = obj.get(color_by, 0)
            rng = (metric_max - metric_min) if metric_max != metric_min else 1.0
            t   = (val - metric_min) / rng
            color = _metric_color(float(t))

        mask_3ch = mask[:, :, np.newaxis]
        overlay  = np.where(mask_3ch, overlay * 0.45 + color * 0.55, overlay)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rgb = (int(color[0]), int(color[1]), int(color[2]))

        cx = int((mask * np.arange(mask.shape[1])[np.newaxis, :]).sum() / (mask.sum() + 1e-6))
        cy = int((mask * np.arange(mask.shape[0])[:, np.newaxis]).sum() / (mask.sum() + 1e-6))
        tmp = overlay.astype(np.uint8)
        cv2.drawContours(tmp, contours, -1, rgb, 2)
        cv2.putText(tmp, str(obj["id"]), (cx - 5, cy + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        overlay = tmp.astype(np.float32)

    return overlay.astype(np.uint8)
