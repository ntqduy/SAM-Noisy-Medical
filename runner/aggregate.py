from typing import Dict
import pandas as pd


GROUP_KEYS = ["phase", "dataset", "model", "weight", "mode", "protocol", "noise", "level"]


def aggregate_results(df: pd.DataFrame) -> pd.DataFrame:
    metrics = ["dice", "iou", "hd95"]
    optional = [c for c in ["pred_iou_score"] if c in df.columns]
    metrics = metrics + optional

    agg = {}
    for m in metrics:
        agg[m] = ["mean", "std", "count"]

    out = df.groupby(GROUP_KEYS, dropna=False).agg(agg).reset_index()
    # flatten columns
    out.columns = ["_".join([c for c in col if c]) if isinstance(col, tuple) else col for col in out.columns]
    return out


def compute_stability(df: pd.DataFrame) -> pd.DataFrame:
    """
    PerfDrop = Dice(L0)-Dice(L4) per (dataset, model, weight, mode, noise, protocol family)
    Assumes L0 exists in P0 or L0 in other cases; we use protocol=P0 for baseline.
    """
    base = df[df["protocol"] == "P0"].copy()
    base = base.rename(columns={"dice": "dice_L0"})[
        ["dataset", "model", "weight", "mode", "id", "dice_L0"]
    ]

    worst = df[(df["protocol"] == "P1") & (df["level"] == "L4")].copy()
    worst = worst.rename(columns={"dice": "dice_L4"})[
        ["dataset", "model", "weight", "mode", "noise", "id", "dice_L4"]
    ]

    merged = worst.merge(base, on=["dataset", "model", "weight", "mode", "id"], how="left")
    merged["perf_drop"] = merged["dice_L0"] - merged["dice_L4"]

    stab = merged.groupby(["dataset", "model", "weight", "mode", "noise"]).agg(
        perf_drop_mean=("perf_drop", "mean"),
        perf_drop_std=("perf_drop", "std"),
        n=("perf_drop", "count"),
    ).reset_index()
    return stab
