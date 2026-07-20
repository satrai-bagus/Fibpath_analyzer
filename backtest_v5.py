"""
Backtest jujur (walk-forward) untuk Fib Pattern Engine V5.
===========================================================
Menghasilkan models/<COIN>/backtest_v5_predictions.csv yang dibaca model_perf.py
untuk mengisi tab "Performa Model" di app.

Skema walk-forward (expanding window, retrain per bulan):
  - Untuk tiap bulan M (mulai `--start-month`, default = bulan ke-7 dataset):
        train  = SEMUA baris dengan dt < awal bulan M      (expanding)
        test   = baris di dalam bulan M
    Model dilatih ulang tiap awal bulan, jadi tidak ada kebocoran masa depan:
    prediksi untuk bulan M hanya memakai data sebelum bulan M.

Definisi kolom (identik dengan model_perf.py):
  fh_actual  = first_hit_target hasil engine (rank fib terkecil > 0)
  dir_actual = arah pemenang di rank terkecil.
               semua pemenang _UP  -> UP
               semua pemenang _DOWN-> DOWN
               campur UP+DOWN, atau tidak ada hit -> TIE
               (baris TIE dibuang saat hitung akurasi arah -> "clear-winner")
  model_fh   = argmax first_hit_probs model (8 kelas, termasuk TIE/NO_HIT)
  model_dir  = UP bila total prob level _UP >= total prob level _DOWN
  conf       = |P(UP) - P(DOWN)|  -> dipakai untuk kurva confidence
  base_*     = baseline "tebak mayoritas": distribusi global data training
               (tanpa melihat fitur baris sama sekali)
  pm_<lvl>   = reach probability model   pb_<lvl> = reach probability baseline
  y_<lvl>    = aktual: level tsb tersentuh dalam 48 jam (1/0)

Pemakaian:
  python backtest_v5.py BTC             # 1 koin
  python backtest_v5.py BTC ETH SOL     # beberapa
  python backtest_v5.py ALL             # semua koin di registry
  python backtest_v5.py BTC --validate models/BTC/backtest_v5_predictions.csv
        # bandingkan hasil dengan CSV lama (uji reproduksibilitas definisi)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from coin_registry import COINS, dataset_path, predictions_path
from fib_pattern_engine_v5 import (
    ACTIONABLE_TARGETS,
    ALL_FIRST_HIT_TARGETS,
    RANK_COLUMNS,
    FibPatternEngineV5,
)

UP_TARGETS = [t for t in ACTIONABLE_TARGETS if t.endswith("_UP")]
DOWN_TARGETS = [t for t in ACTIONABLE_TARGETS if t.endswith("_DOWN")]

WARMUP_MONTHS = 6  # bulan pertama dipakai murni sebagai data training awal
LOOKAHEAD_H = 48   # jendela lookahead dataset; lihat EMBARGO di bawah

# Embargo anti-kebocoran: label sebuah bar baru final setelah LOOKAHEAD_H jam ke
# depan. Bar pada 30 Juni 23:00 hasilnya ditentukan harga sampai 2 Juli — sudah
# masuk periode uji. Karena itu 48 jam terakhir sebelum bulan uji DIBUANG dari
# training, bukan cuma dipotong di batas bulan.


# ------------------------------------------------------------------ helpers
def actual_direction(row: pd.Series) -> str:
    """Arah sebenarnya: siapa yang tersentuh lebih dulu (rank terkecil)."""
    ranks = {t: int(row[c]) for t, c in RANK_COLUMNS.items()}
    positive = {t: r for t, r in ranks.items() if r > 0}
    if not positive:
        return "TIE"  # NO_HIT_48H
    lowest = min(positive.values())
    winners = [t for t, r in positive.items() if r == lowest]
    up = any(t.endswith("_UP") for t in winners)
    down = any(t.endswith("_DOWN") for t in winners)
    if up and not down:
        return "UP"
    if down and not up:
        return "DOWN"
    return "TIE"  # tersentuh dua arah pada jam yang sama


def direction_from_probs(probs: dict) -> tuple[str, float]:
    """Arah + confidence dari distribusi first-hit."""
    p_up = sum(float(probs.get(t, 0.0)) for t in UP_TARGETS)
    p_down = sum(float(probs.get(t, 0.0)) for t in DOWN_TARGETS)
    return ("UP" if p_up >= p_down else "DOWN"), abs(p_up - p_down)


def argmax_target(probs: dict) -> str:
    return max(ALL_FIRST_HIT_TARGETS, key=lambda t: float(probs.get(t, 0.0)))


def load_frames(xlsx: Path):
    """Return (raw_df_with_dt, prepared_df_with_dt).

    raw dipakai untuk melatih (fit_dataframe menormalkan sendiri), prepared
    dipakai untuk membaca fitur + label aktual per baris uji.
    """
    eng = FibPatternEngineV5()
    raw = eng._load_workbook(xlsx, "ALL")
    dates = raw["Date"].apply(eng._parse_indonesian_date)
    hours = pd.to_numeric(raw["Clock"], errors="coerce")
    raw = raw.copy()
    raw["__dt"] = pd.to_datetime(dates) + pd.to_timedelta(hours, unit="h")

    prepared = eng._prepare_dataframe(raw)  # kolom __dt otomatis terbuang
    prepared["dt"] = pd.to_datetime(prepared["Date"]) + pd.to_timedelta(
        prepared["Clock"], unit="h"
    )
    prepared = prepared.sort_values("dt").reset_index(drop=True)
    return raw, prepared


# ------------------------------------------------------------------ backtest
def backtest_coin(coin: str, start_month: str | None = None,
                  dataset: str | Path | None = None,
                  max_months: int | None = None) -> pd.DataFrame:
    xlsx = Path(dataset) if dataset else dataset_path(coin)
    if not xlsx.exists():
        raise FileNotFoundError(f"Dataset tidak ada: {xlsx}")

    raw, prepared = load_frames(xlsx)
    months = prepared["dt"].dt.to_period("M")
    all_months = sorted(months.unique())

    if start_month:
        first = pd.Period(start_month, "M")
    else:
        if len(all_months) <= WARMUP_MONTHS:
            raise ValueError(
                f"{coin}: dataset cuma {len(all_months)} bulan, "
                f"butuh > {WARMUP_MONTHS} bulan untuk walk-forward."
            )
        first = all_months[WARMUP_MONTHS]

    test_months = [m for m in all_months if m >= first]
    if max_months:
        test_months = test_months[:max_months]
    rows: list[dict] = []

    for mi, per in enumerate(test_months, 1):
        m_start = per.start_time
        m_end = per.end_time

        embargo = m_start - pd.Timedelta(hours=LOOKAHEAD_H)
        train_raw = raw[raw["__dt"] < embargo]
        test = prepared[(prepared["dt"] >= m_start) & (prepared["dt"] <= m_end)]
        if test.empty or train_raw.empty:
            continue

        engine = FibPatternEngineV5().fit_dataframe(train_raw)

        # baseline: distribusi global data training, tidak melihat fitur baris
        base_fh = argmax_target(engine.global_first_hit_probs)
        base_dir, _ = direction_from_probs(engine.global_first_hit_probs)
        base_reach = engine.global_reach_probs

        n_train = len(engine.train_df)
        print(f"  [{coin}] {per}  train={n_train:>6}  test={len(test):>4}"
              f"  ({mi}/{len(test_months)})", flush=True)

        for _, r in test.iterrows():
            setup = {
                "Trend": r["Trend"],
                "SQZMOM 1 Momentum": r["SQZMOM 1 Momentum"],
                "SQZMOM 1 Squeeze": r["SQZMOM 1 Squeeze"],
                "SQZMOM 2 Momentum": r["SQZMOM 2 Momentum"],
                "SQZMOM 2 Squeeze": r["SQZMOM 2 Squeeze"],
                "Bar 1": r["Bar 1"],
                "Bar 2": r["Bar 2"],
                "Raw Position": r["Raw Position"],
                "Final Position": r["Final Position"],
                "Score": r["Score"],
                "Last TR": r["Last TR"],
                "SQZMOM 1 Value": r["SQZMOM 1 Value"],
                "SQZMOM 2 Value": r["SQZMOM 2 Value"],
            }
            res = engine.predict(setup, top_k_matches=5)
            model_dir, conf = direction_from_probs(res.first_hit_probs)

            rec = {
                "dt": r["dt"],
                "fh_actual": r["first_hit_target"],
                "dir_actual": actual_direction(r),
                "model_fh": argmax_target(res.first_hit_probs),
                "base_fh": base_fh,
                "model_dir": model_dir,
                "base_dir": base_dir,
                "conf": conf,
                "trend": r["Trend"],
                "raw": r["Raw Position"],
            }
            for lvl in ACTIONABLE_TARGETS:
                rec[f"pm_{lvl}"] = float(res.reach_probs.get(lvl, 0.0))
                rec[f"pb_{lvl}"] = float(base_reach.get(lvl, 0.0))
                rec[f"y_{lvl}"] = int(bool(r[f"reach_{lvl}"]))
            rows.append(rec)

    return pd.DataFrame(rows)


# ------------------------------------------------------------------ validate
def validate(new: pd.DataFrame, ref_path: Path) -> None:
    """Bandingkan hasil dengan CSV referensi (uji apakah definisi kita persis sama)."""
    ref = pd.read_csv(ref_path)
    ref["dt"] = pd.to_datetime(ref["dt"])
    new = new.copy()
    new["dt"] = pd.to_datetime(new["dt"])
    m = ref.merge(new, on="dt", suffixes=("_ref", "_new"))
    print(f"\n  baris beririsan: {len(m)} (ref={len(ref)}, baru={len(new)})")
    if not len(m):
        print("  !! tidak ada dt yang beririsan"); return

    exact = ["fh_actual", "dir_actual", "model_fh", "base_fh", "model_dir", "base_dir",
             "trend", "raw"]
    print("  --- kolom kategorikal (target: 1.000) ---")
    for c in exact:
        if f"{c}_ref" in m:
            print(f"    {c:<12} {(m[f'{c}_ref'] == m[f'{c}_new']).mean():.4f}")

    print("  --- kolom numerik (max selisih absolut) ---")
    for c in ["conf"] + [f"pm_{l}" for l in ACTIONABLE_TARGETS] + \
             [f"pb_{l}" for l in ACTIONABLE_TARGETS]:
        if f"{c}_ref" in m:
            d = (m[f"{c}_ref"] - m[f"{c}_new"]).abs().max()
            print(f"    {c:<14} {d:.2e}")


# ------------------------------------------------------------------ main
def main() -> None:
    ap = argparse.ArgumentParser(description="Walk-forward backtest Fib Engine V5")
    ap.add_argument("coins", nargs="*", default=["ALL"])
    ap.add_argument("--start-month", default=None,
                    help="bulan pertama yang di-backtest, YYYY-MM "
                         f"(default: bulan ke-{WARMUP_MONTHS + 1} dataset)")
    ap.add_argument("--validate", default=None,
                    help="path CSV referensi untuk dibandingkan (tidak menulis output)")
    ap.add_argument("--dataset", default=None,
                    help="override path dataset .xlsx (untuk validasi ke dataset lama)")
    ap.add_argument("--max-months", type=int, default=None,
                    help="batasi jumlah bulan yang di-backtest (uji cepat)")
    ap.add_argument("--out", default=None, help="override path output CSV")
    args = ap.parse_args()

    coins = list(COINS) if (not args.coins or args.coins[0].upper() == "ALL") \
        else [c.upper() for c in args.coins]

    for coin in coins:
        if coin not in COINS:
            print(f"! {coin} tidak ada di registry, lewati."); continue
        print(f"\n=== Backtest {coin} ===", flush=True)
        try:
            R = backtest_coin(coin, args.start_month, args.dataset, args.max_months)
        except Exception as e:
            print(f"! {coin} gagal: {type(e).__name__}: {e}"); continue
        if R.empty:
            print(f"! {coin}: tidak ada baris hasil."); continue

        if args.validate:
            validate(R, Path(args.validate))
            continue

        out = Path(args.out) if args.out else predictions_path(coin)
        R.to_csv(out, index=False)
        clear = R[R["dir_actual"] != "TIE"]
        acc = (clear["model_dir"] == clear["dir_actual"]).mean()
        base = (clear["base_dir"] == clear["dir_actual"]).mean()
        print(f"  -> {out}  ({len(R)} baris, {R.dt.min()} .. {R.dt.max()})")
        print(f"     akurasi arah {acc:.1%} vs baseline {base:.1%} "
              f"(edge {acc - base:+.1%}) pada {len(clear)} baris clear-winner")


if __name__ == "__main__":
    main()
