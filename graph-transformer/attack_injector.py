"""Synthetic timing-disrupting attack injector.

Writes short attack captures cut from each ambient ROAD capture into outputs/synthetic/.
Leakage rule: a synthetic capture is only ever evaluated in the split/fold where its parent
ambient capture is held out for test (see train.is_test_graph), so injected data never
reaches training, validation or threshold calibration. Intensity levels are fixed below and
must not be tuned by looking at detection results.
"""
import os
import glob
import json
import numpy as np
import pandas as pd

from preprocess import AMBIENT_DIR, SYNTHETIC_DIR

SEED = 42
EPISODE_SEC = 10.0
PAD_SEC = 2.0                      # benign context before/after the episode (one window length)
TARGET_IDS = [208, 1760, 1255]     # ROAD-documented injection targets (0x0D0, 0x6E0, 0x4E7)
FLOOD_ID = 0                       # highest-priority ID; absent from ROAD -> <UNK> in the vocabulary

LEVELS = {
    "fabrication": [1, 2, 4],          # extra spoofed frames per legitimate frame (legit frames kept)
    "suspension": [0.5, 0.9, 1.0],     # fraction of the target's frames removed
    "drift": [0.01, 0.03, 0.10],       # attacker period error vs. the real ECU (frames replaced)
    "flooding": [50, 200, 1000],       # frames per second of FLOOD_ID
}


def _spoof_row(template: pd.Series, spoof_values: pd.Series, t: float) -> dict:
    row = template.to_dict()
    row.update(spoof_values.to_dict())
    row["Time"] = t
    row["Label"] = 1
    return row


def inject(df: pd.DataFrame, attack: str, level: float, target: int, rng: np.random.Generator,
           start: float = PAD_SEC, end: float = PAD_SEC + EPISODE_SEC) -> pd.DataFrame:
    """Apply one attack episode to `df` (Time already rebased) within [start, end)."""
    df = df.copy()
    signal_cols = [c for c in df.columns if c.startswith("Signal_")]
    in_ep = (df["Time"] >= start) & (df["Time"] < end)
    tgt = df["ID"] == target
    legit = df[in_ep & tgt]
    period = float(np.median(np.diff(df.loc[tgt, "Time"].values))) if tgt.sum() > 1 else 0.01
    spoof = df.loc[tgt, signal_cols].max()  # "max override" payload, as in ROAD's max-value attacks
    new_rows = []

    if attack == "fabrication":
        for _, r in legit.iterrows():
            for j in range(1, int(level) + 1):
                new_rows.append(_spoof_row(r, spoof, r["Time"] + j * period / (level + 1)))
    elif attack == "suspension":
        drop = rng.choice(legit.index.values, size=int(round(level * len(legit))), replace=False)
        df = df.drop(index=drop)
    elif attack == "drift":
        df = df.drop(index=legit.index)
        if len(legit):
            t, template = float(legit["Time"].iloc[0]), legit.iloc[0]
            while t < end:
                new_rows.append(_spoof_row(template, spoof, t))
                t += period * (1 + level) + rng.normal(0, 0.05 * period)
    elif attack == "flooding":
        for t in np.sort(rng.uniform(start, end, int(level * (end - start)))):
            row = {c: np.nan for c in df.columns}
            row.update({"Time": t, "ID": FLOOD_ID, "Label": 1})
            new_rows.append(row)
    else:
        raise ValueError(f"Unknown attack type: {attack}")

    if new_rows:
        df = pd.concat([df, pd.DataFrame(new_rows, columns=df.columns)], ignore_index=True)
    # iterrows() upcasts template rows to float; IDs must stay integers or they no longer
    # match the vocabulary ("208.0" != "208") and every node becomes <UNK>.
    df["ID"] = df["ID"].astype("int64")
    df["Label"] = df["Label"].astype("int64")
    return df.sort_values("Time", kind="stable").reset_index(drop=True)


def main():
    os.makedirs(SYNTHETIC_DIR, exist_ok=True)
    for old in glob.glob(os.path.join(SYNTHETIC_DIR, "*.csv")):
        os.remove(old)
    rng = np.random.default_rng(SEED)
    metadata = {}
    parents = sorted(glob.glob(os.path.join(AMBIENT_DIR, "ambient_*.csv")))
    for p_idx, path in enumerate(parents):
        parent = os.path.splitext(os.path.basename(path))[0]
        full = pd.read_csv(path)
        full = full.sort_values("Time", kind="stable").reset_index(drop=True)
        t_min, t_max = full["Time"].min(), full["Time"].max()
        print(f"{parent}: {t_max - t_min:.0f}s")
        for a_idx, (attack, levels) in enumerate(LEVELS.items()):
            target = FLOOD_ID if attack == "flooding" else TARGET_IDS[(p_idx + a_idx) % len(TARGET_IDS)]
            for level in levels:
                ep_start = t_min + rng.uniform(0.3, 0.6) * (t_max - t_min - EPISODE_SEC - 2 * PAD_SEC) + PAD_SEC
                crop = full[(full["Time"] >= ep_start - PAD_SEC) & (full["Time"] < ep_start + EPISODE_SEC + PAD_SEC)].copy()
                crop["Time"] = crop["Time"] - (ep_start - PAD_SEC)
                out = inject(crop, attack, level, target, rng)
                name = f"synthetic__{parent}__{attack}_{level}"
                out.to_csv(os.path.join(SYNTHETIC_DIR, f"{name}.csv"), index=False)
                metadata[name] = {
                    "synthetic": True,
                    "parent": parent,
                    "attack": attack,
                    "level": level,
                    "target_id": "<UNK>" if attack == "flooding" else str(target),
                    "injection_interval": [PAD_SEC, PAD_SEC + EPISODE_SEC],
                    "parent_offset_sec": float(ep_start - PAD_SEC - t_min),
                    "frames_added": int(len(out) - len(crop)) if len(out) >= len(crop) else 0,
                    "frames_removed": int(len(crop) - len(out)) if len(out) < len(crop) else 0,
                }
    with open(os.path.join(SYNTHETIC_DIR, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Wrote {len(metadata)} synthetic captures to {SYNTHETIC_DIR}")


if __name__ == "__main__":
    main()
