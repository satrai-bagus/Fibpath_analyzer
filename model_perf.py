"""
Hitung metrik performa model dari hasil backtest jujur (backtest_v5_predictions.csv).
Semua angka dihitung ulang dari data, bukan hardcode — jadi selalu sinkron dgn model.

Definisi (identik dengan backtest_v5.py):
  - Baris "clear-winner" = dir_actual != TIE.
  - Direction accuracy = P(model_dir == dir_actual) pada clear-winner.
  - Edge vs base = direction acc - akurasi prediksi-mayoritas (base_dir).
  - Confident-decile = akurasi pada 10% conf tertinggi.
  - First-hit 8-kelas = P(model_fh == fh_actual) seluruh baris.
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd


def compute_perf(pred_path: str | Path) -> Optional[Dict[str, object]]:
    pred_path = Path(pred_path)
    if not pred_path.exists():
        return None
    R = pd.read_csv(pred_path)
    if R.empty:
        return None

    R["dt"] = pd.to_datetime(R["dt"], errors="coerce")
    clear = R[R["dir_actual"] != "TIE"].copy()
    n = len(clear)
    if n == 0:
        return None

    dir_acc = float((clear["model_dir"] == clear["dir_actual"]).mean())
    base_acc = float((clear["base_dir"] == clear["dir_actual"]).mean())
    se = (dir_acc * (1 - dir_acc) / n) ** 0.5 if n else float("nan")
    z = (dir_acc - 0.5) / se if se else float("nan")

    # confident decile
    thr = clear["conf"].quantile(0.9)
    top = clear[clear["conf"] >= thr]
    conf_acc = float((top["model_dir"] == top["dir_actual"]).mean()) if len(top) else float("nan")

    # naive "follow trend"
    naive_trend = np.where(clear["trend"].astype(str).str.lower() == "long", "UP", "DOWN")
    naive_acc = float((naive_trend == clear["dir_actual"].values).mean())

    # first-hit 8-class
    fh_acc = float((R["model_fh"] == R["fh_actual"]).mean())
    fh_base = float((R["base_fh"] == R["fh_actual"]).mean())

    # per-month direction accuracy
    clear["month"] = clear["dt"].dt.to_period("M").astype(str)
    per_month = (
        clear.groupby("month")
        .apply(lambda g: (g["model_dir"] == g["dir_actual"]).mean(), include_groups=False)
        .rename("acc")
        .reset_index()
    )
    months_win = int((per_month["acc"] > 0.5).sum())
    months_total = int(len(per_month))

    # confidence curve: accuracy at increasing confidence cutoffs
    curve = []
    for q in [0.0, 0.5, 0.75, 0.9, 0.95]:
        t = clear["conf"].quantile(q)
        sub = clear[clear["conf"] >= t]
        if len(sub):
            curve.append({"top_frac": round(1 - q, 2),
                          "acc": float((sub["model_dir"] == sub["dir_actual"]).mean()),
                          "n": int(len(sub))})

    return {
        "n_clear": n,
        "n_total": int(len(R)),
        "dir_acc": dir_acc,
        "base_acc": base_acc,
        "edge": dir_acc - base_acc,
        "z": z,
        "conf_acc": conf_acc,
        "naive_trend_acc": naive_acc,
        "edge_vs_naive": dir_acc - naive_acc,
        "fh_acc": fh_acc,
        "fh_base": fh_base,
        "fh_edge": fh_acc - fh_base,
        "months_win": months_win,
        "months_total": months_total,
        "per_month": per_month,
        "conf_curve": pd.DataFrame(curve),
        "date_min": R["dt"].min(),
        "date_max": R["dt"].max(),
    }


if __name__ == "__main__":
    import sys
    from coin_registry import predictions_path, available_coins
    coins = sys.argv[1:] or available_coins()
    rows = []
    for c in coins:
        p = compute_perf(predictions_path(c))
        if p:
            rows.append({
                "coin": c, "dir_acc": round(p["dir_acc"], 3), "edge": round(p["edge"], 3),
                "z": round(p["z"], 1), "conf10": round(p["conf_acc"], 3),
                "fh_edge": round(p["fh_edge"], 3), "months": f"{p['months_win']}/{p['months_total']}",
                "n": p["n_clear"],
            })
    print(pd.DataFrame(rows).to_string(index=False))
