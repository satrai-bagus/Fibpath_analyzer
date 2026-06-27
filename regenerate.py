"""
Regenerate dataset + model untuk satu koin (atau semua) ke models/<COIN>/.

Contoh:
  python regenerate.py BTC          # 1 koin
  python regenerate.py BTC ETH SOL  # beberapa
  python regenerate.py ALL          # semua koin di registry

Yang dilakukan per koin:
  1. fetch dataset SUIUSDT-style dari Binance -> models/<COIN>/Dataset Output Otomatis <COIN> 2025-2026.xlsx
  2. latih engine k-NN -> models/<COIN>/fib_pattern_engine_v5.pkl (+ summary csv)

Catatan: untuk refresh angka tab "Performa Model", jalankan backtest jujur juga
(salin backtest_v5.py per-koin lama, atau panggil engine pada walk-forward) lalu
taruh backtest_v5_predictions.csv di models/<COIN>/.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from coin_registry import COINS, coin_dir, dataset_path, model_path
from fib_pattern_engine_v5 import train_and_save_model_v5

BASE = Path(__file__).resolve().parent
START_MONTH = "2025-01"


def regen(coin: str) -> None:
    coin = coin.upper()
    if coin not in COINS:
        print(f"  ! {coin} tidak ada di registry, lewati."); return
    cdir = coin_dir(coin)
    cdir.mkdir(parents=True, exist_ok=True)
    sym = COINS[coin]["symbol"]
    out_xlsx = dataset_path(coin)

    print(f"[{coin}] 1/2 fetch dataset {sym} -> {out_xlsx.name}")
    cmd = [sys.executable, str(BASE / "fib_dataset_export.py"),
           "--symbol", sym, "--start", START_MONTH, "--out", str(out_xlsx)]
    subprocess.run(cmd, check=True)

    print(f"[{coin}] 2/2 train engine -> {model_path(coin).name}")
    train_and_save_model_v5(
        excel_path=out_xlsx,
        model_path=model_path(coin),
        first_hit_summary_csv=cdir / "fib_pattern_first_hit_summary_v5.csv",
        reach_summary_csv=cdir / "fib_pattern_reach_summary_v5.csv",
    )
    print(f"[{coin}] DONE\n")


def main() -> None:
    args = sys.argv[1:]
    coins = list(COINS) if (not args or args[0].upper() == "ALL") else [a.upper() for a in args]
    print(f"Regenerate: {', '.join(coins)}")
    for c in coins:
        regen(c)


if __name__ == "__main__":
    main()
