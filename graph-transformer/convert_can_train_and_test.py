"""Convert one can-train-and-test set (DTU, Lampe & Meng 2023) into ROAD's CSV layout.

Input  : the second/can-train-and-test/<set>/{train_01,test_0k_*}/*.csv
         columns: timestamp, arbitration_id (hex), data_field (hex payload), attack (0/1)
Output : the second/converted/ctt_<set>/{ambient,attacks}/*.csv
         columns: Label, Time, ID (int), Signal_1_of_ID .. Signal_8_of_ID (payload byte k, NaN past DLC)

Protocol (unsupervised, like ROAD): only train_01/attack-free-*.csv is benign training data. Each is
cut into AMBIENT_CHUNKS equal-time parts so the capture-level split has enough captures; parts of the
same drive can land in train and val/test, which is weaker than ROAD's drive-level split. The dataset's
own test_01..test_04 files (benign + attack frames) are the test set. Labelled attack files under
train_01 are left out: they are meant for supervised training.

Run: python convert_can_train_and_test.py set_01   then   CAN_DATASET=ctt_set_01 python preprocess.py ...
"""
import os
import sys
import glob
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SECOND_DIR = os.path.join(os.path.dirname(BASE_DIR), "the second")
RAW_DIR = os.path.join(SECOND_DIR, "can-train-and-test")
AMBIENT_CHUNKS = 4
N_BYTES = 8

# ASCII -> nibble lookup; anything that is not a hex digit (the pad character) maps to -1.
_NIBBLE = np.full(256, -1, dtype=np.int16)
for _i, _c in enumerate("0123456789abcdef"):
    _NIBBLE[ord(_c)] = _i
    _NIBBLE[ord(_c.upper())] = _i


def payload_bytes(data_field: pd.Series) -> np.ndarray:
    """Hex payload strings -> (n, 8) float array of byte values, NaN beyond each frame's DLC."""
    padded = data_field.fillna("").astype(str).str.slice(0, 2 * N_BYTES).str.ljust(2 * N_BYTES, "z")
    chars = np.frombuffer("".join(padded).encode("ascii"), dtype=np.uint8).reshape(-1, N_BYTES, 2)
    hi, lo = _NIBBLE[chars[..., 0]], _NIBBLE[chars[..., 1]]
    out = (hi * 16 + lo).astype(np.float32)
    out[(hi < 0) | (lo < 0)] = np.nan
    return out


def convert_frame(df: pd.DataFrame) -> pd.DataFrame:
    ids = df["arbitration_id"].astype(str)
    id_map = {h: int(h, 16) for h in ids.unique()}
    out = pd.DataFrame({
        "Label": df["attack"].astype(np.int64).to_numpy(),
        "Time": df["timestamp"].astype(np.float64).to_numpy(),
        "ID": ids.map(id_map).astype(np.int64).to_numpy(),
    })
    signals = payload_bytes(df["data_field"])
    for k in range(N_BYTES):
        out[f"Signal_{k + 1}_of_ID"] = signals[:, k]
    return out


def main(set_name: str):
    out_root = os.path.join(SECOND_DIR, "converted", f"ctt_{set_name}")
    amb_dir, att_dir = os.path.join(out_root, "ambient"), os.path.join(out_root, "attacks")
    os.makedirs(amb_dir, exist_ok=True)
    os.makedirs(att_dir, exist_ok=True)
    src = os.path.join(RAW_DIR, set_name)

    for path in sorted(glob.glob(os.path.join(src, "train_01", "attack-free-*.csv"))):
        df = convert_frame(pd.read_csv(path)).sort_values("Time", kind="stable")
        if df["Label"].any():
            raise ValueError(f"{path} is labelled attack-free but has attack frames")
        edges = np.linspace(df["Time"].min(), df["Time"].max(), AMBIENT_CHUNKS + 1)
        stem = os.path.splitext(os.path.basename(path))[0]
        for c in range(AMBIENT_CHUNKS):
            part = df[(df["Time"] >= edges[c]) & ((df["Time"] < edges[c + 1]) | (c == AMBIENT_CHUNKS - 1))]
            part.to_csv(os.path.join(amb_dir, f"ambient_{stem}_part{c + 1}.csv"), index=False)
        print(f"ambient  {stem}: {len(df):,} frames, {df['Time'].max() - df['Time'].min():.0f}s -> {AMBIENT_CHUNKS} parts")

    for path in sorted(glob.glob(os.path.join(src, "test_0*", "*.csv"))):
        subset = os.path.basename(os.path.dirname(path))[:7]  # test_01 .. test_04
        stem = os.path.splitext(os.path.basename(path))[0]
        df = convert_frame(pd.read_csv(path))
        df.to_csv(os.path.join(att_dir, f"{subset}__{stem}.csv"), index=False)
        print(f"attack   {subset}/{stem}: {len(df):,} frames, {df['Label'].mean():.1%} attack frames")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "set_01")
