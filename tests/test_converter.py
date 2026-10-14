import os
import sys
import unittest
import numpy as np
import pandas as pd

GT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "graph-transformer")
if GT_DIR not in sys.path:
    sys.path.insert(0, GT_DIR)

from convert_can_train_and_test import convert_frame, benign_segments


class TestConverter(unittest.TestCase):
    def test_bytes_ids_labels_and_short_dlc(self):
        raw = pd.DataFrame({
            "timestamp": [1.0, 1.5, 2.0],
            "arbitration_id": ["0F1", "1A0", "0F1"],
            "data_field": ["28030040", "FF00ab12cd34ef56", np.nan],
            "attack": [0, 1, 0],
        })
        out = convert_frame(raw)
        self.assertEqual(out["ID"].tolist(), [0xF1, 0x1A0, 0xF1])
        self.assertEqual(out["Label"].tolist(), [0, 1, 0])
        self.assertEqual(out.iloc[0, 3:7].tolist(), [0x28, 0x03, 0x00, 0x40])
        self.assertTrue(out.iloc[0, 7:].isna().all())          # DLC 4: bytes 5-8 are NaN
        self.assertEqual(out.iloc[1, 3:].tolist(), [255, 0, 0xAB, 0x12, 0xCD, 0x34, 0xEF, 0x56])
        self.assertTrue(out.iloc[2, 3:].isna().all())          # empty payload

    def test_benign_segments_keep_margin_from_attacks(self):
        t = np.arange(0, 200, 0.01)
        df = pd.DataFrame({"Time": t, "Label": ((t > 100) & (t < 101)).astype(int)})
        segs = benign_segments(df)
        self.assertEqual(len(segs), 2)
        for seg in segs:
            self.assertEqual(int(seg["Label"].sum()), 0)
            self.assertTrue(((seg["Time"] < 98.02) | (seg["Time"] > 102.98)).all())


if __name__ == "__main__":
    unittest.main()
