# ROAD Dataset Preprocessing

import os
import glob
import json
import pickle
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


# 1. CONFIGURATION
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROAD_ROOT = os.path.join(BASE_DIR, "road", "road")

# Signal-translated captures live under these subfolders in the ROAD release.
AMBIENT_DIR = os.path.join(ROAD_ROOT, "signal_extractions", "ambient")
ATTACK_DIR = os.path.join(ROAD_ROOT, "signal_extractions", "attacks")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

# Windowing parameters
# These were intentionally chosen to be conservative and stable for the ROAD data.
# Raising them greatly increases the amount of raw message data retained per window.
WINDOW_SIZE_SEC = 2.0     # length of each temporal window, in seconds
STRIDE_SEC_BENIGN = 1.0   # allow overlap for benign windows (more training density)
STRIDE_SEC_ATTACK = 2.0   # non-overlapping for attack windows (avoid train/test leakage)
MIN_MESSAGES_PER_WINDOW = 5  # minimum number of CAN messages needed for a window to be kept

# The graph-building stage expects each window to retain the raw message rows.
# Set True: graph pipeline can run.
# Set False: pure statistics-only pass and want to save RAM
KEEP_RAW_WINDOW_MESSAGES = True


# 2. DATA STRUCTURES

@dataclass
class CaptureFile:
    """Represents one raw ROAD CSV capture before windowing."""
    path: str
    capture_name: str
    is_attack: bool
    metadata: dict = field(default_factory=dict)


@dataclass
class WindowRecord:

    capture_name: str
    window_start: float
    window_end: float
    messages: Optional[pd.DataFrame] = None  # raw rows within this window, only when explicitly enabled
    label: int = 0                          # 0 = benign window, 1 = contains attack traffic
    frac_attack_messages: float = 0.0


# 3. FILE DISCOVERY

def discover_captures(ambient_dir: str, attack_dir: str) -> list[CaptureFile]:

    captures = []

    for csv_path in sorted(glob.glob(os.path.join(ambient_dir, "*.csv"))):
        filename = os.path.basename(csv_path)
        if not filename.startswith("ambient_"):
            continue
        name = os.path.splitext(filename)[0]
        captures.append(CaptureFile(
            path=csv_path,
            capture_name=name,
            is_attack=False,
            metadata=_load_metadata_if_exists(csv_path),
        ))

    for csv_path in sorted(glob.glob(os.path.join(attack_dir, "*.csv"))):
        name = os.path.splitext(os.path.basename(csv_path))[0]
        captures.append(CaptureFile(
            path=csv_path,
            capture_name=name,
            is_attack=True,
            metadata=_load_metadata_if_exists(csv_path),
        ))

    if not captures:
        raise FileNotFoundError(
            f"No CSV captures found under {ambient_dir} or {attack_dir}. "
            f"Check ROAD_ROOT / AMBIENT_DIR / ATTACK_DIR paths."
        )

    return captures


def _load_metadata_if_exists(csv_path: str) -> dict:
  
    candidates = [
        os.path.join(os.path.dirname(csv_path), "metadata.json"),
        csv_path.replace(".csv", "_metadata.json"),
    ]

    for candidate in candidates:
        if os.path.exists(candidate):
            with open(candidate, "r") as f:
                return json.load(f)
    return {}


# 4. CSV PARSING

def load_capture(capture: CaptureFile) -> pd.DataFrame:

    df = pd.read_csv(capture.path)

    expected_cols = {"Label", "ID", "Time"}
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"{capture.path} is missing expected columns: {missing}. "
            f"Confirm this is a ROAD signal-translated CSV."
        )

    # Normalize dtypes
    df["Label"] = df["Label"].astype(int)
    df["ID"] = df["ID"].astype(str)  # keep as string/hex-safe; avoid int overflow surprises
    df["Time"] = df["Time"].astype(float)
    df = df.sort_values("Time").reset_index(drop=True) # Safety check 

    # Re-baseline time to start at 0 for this capture, simpler windowing math
    df["Time"] = df["Time"] - df["Time"].min()

    return df



# 5. TEMPORAL WINDOWING


def window_capture(df: pd.DataFrame, capture_name: str, is_attack: bool, keep_messages: bool = KEEP_RAW_WINDOW_MESSAGES) -> list[WindowRecord]:
    stride = STRIDE_SEC_BENIGN if not is_attack else STRIDE_SEC_ATTACK
    total_duration = df["Time"].max()

    windows = []
    window_start = 0.0

    while window_start < total_duration:
        window_end = window_start + WINDOW_SIZE_SEC

        mask = (df["Time"] >= window_start) & (df["Time"] < window_end)
        window_df = df.loc[mask]

        if len(window_df) >= MIN_MESSAGES_PER_WINDOW:
            frac_attack = float(window_df["Label"].mean())
            # Window-level label: any attack-labeled message -> window is "attack"
            window_label = int(window_df["Label"].max())

            windows.append(WindowRecord(
                capture_name=capture_name,
                window_start=window_start,
                window_end=window_end,
                messages=window_df.reset_index(drop=True) if keep_messages else None,
                label=window_label,
                frac_attack_messages=frac_attack,
            ))

        window_start += stride

    return windows

# 5. DRIVER CODE

def build_dataset(keep_messages: bool = KEEP_RAW_WINDOW_MESSAGES) -> list[WindowRecord]:
    captures = discover_captures(AMBIENT_DIR, ATTACK_DIR)
    print(f"Discovered {len(captures)} captures "
          f"({sum(not c.is_attack for c in captures)} ambient, "
          f"{sum(c.is_attack for c in captures)} attack).")

    all_windows: list[WindowRecord] = []

    for capture in captures:
        print(f"  Processing {capture.capture_name} "
              f"({'attack' if capture.is_attack else 'ambient'}) ...")

        df = load_capture(capture)
        windows = window_capture(df, capture.capture_name, capture.is_attack, keep_messages=keep_messages)
        all_windows.extend(windows)

        print(f"    -> {len(df)} messages, {len(windows)} windows kept "
              f"(min {MIN_MESSAGES_PER_WINDOW} msgs/window)")

    n_benign = sum(w.label == 0 for w in all_windows)
    n_attack = sum(w.label == 1 for w in all_windows)
    print(f"\nTotal windows: {len(all_windows)} "
          f"({n_benign} benign, {n_attack} attack)")

    return all_windows


def save_dataset(windows: list[WindowRecord], output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "road_windowed.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(windows, f)
    print(f"Saved {len(windows)} windows to {out_path}")

    # Persist windowing parameters alongside dataset
    meta_path = os.path.join(output_dir, "road_windowed_metadata.json")
    metadata = {
        "window_duration_sec": WINDOW_SIZE_SEC,
        "benign_stride_sec": STRIDE_SEC_BENIGN,
        "attack_stride_sec": STRIDE_SEC_ATTACK,
        "min_messages_per_window": MIN_MESSAGES_PER_WINDOW,
        "total_windows": len(windows),
        "benign_windows": sum(w.label == 0 for w in windows),
        "attack_windows": sum(w.label == 1 for w in windows),
    }
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved windowing metadata to {meta_path}")


if __name__ == "__main__":
    windows = build_dataset()
    save_dataset(windows, OUTPUT_DIR)
