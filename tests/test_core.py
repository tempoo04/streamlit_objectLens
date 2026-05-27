import unittest

import numpy as np

from core.features import compute_features
from core.overlay import render_overlay


class CoreFeatureTests(unittest.TestCase):
    def test_compute_features_uses_orientation_independent_aspect_ratio(self):
        mask = np.zeros((20, 20), dtype=np.float32)
        mask[2:14, 5:9] = 1.0
        detections = [
            {
                "id": 1,
                "category": "object",
                "score": 0.99,
                "box": [5, 2, 9, 14],
                "mask_raw": mask,
                "mask_thresh": 0.5,
            }
        ]

        df, objects = compute_features(detections)

        self.assertEqual(len(df), 1)
        self.assertEqual(len(objects), 1)
        self.assertGreaterEqual(df.loc[0, "aspect_ratio"], 1.0)
        self.assertEqual(df.loc[0, "area_px"], 48)

    def test_render_overlay_preserves_image_shape(self):
        img = np.zeros((12, 12, 3), dtype=np.uint8)
        mask = np.zeros((12, 12), dtype=np.uint8)
        mask[3:9, 3:9] = 1
        objects = [{"id": 1, "mask": mask, "roundness": 0.8}]

        overlay = render_overlay(img, objects, "roundness")

        self.assertEqual(overlay.shape, img.shape)
        self.assertEqual(overlay.dtype, np.uint8)
        self.assertGreater(int(overlay.sum()), 0)


if __name__ == "__main__":
    unittest.main()
