"""
Registry semua koin yang didukung Fib Path Analyzer V5 (Multi-Coin).

Setiap koin punya model + dataset + backtest SENDIRI di models/<COIN>/.
Tidak ada model yang digabung — switching koin = load .pkl koin tsb.
Engine analisa (fib_pattern_engine_v5) identik untuk semua, tidak diubah.
"""
from __future__ import annotations
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

# Urut sesuai kapitalisasi pasar (kira-kira). emoji + accent = identitas visual UI.
COINS = {
    "BTC":  {"name": "Bitcoin",  "ticker": "BTC-USD",  "symbol": "BTCUSDT",  "emoji": "🟧", "accent": "#F7931A"},
    "ETH":  {"name": "Ethereum", "ticker": "ETH-USD",  "symbol": "ETHUSDT",  "emoji": "🔷", "accent": "#627EEA"},
    "BNB":  {"name": "BNB",      "ticker": "BNB-USD",  "symbol": "BNBUSDT",  "emoji": "🟨", "accent": "#F0B90B"},
    "SOL":  {"name": "Solana",   "ticker": "SOL-USD",  "symbol": "SOLUSDT",  "emoji": "🟣", "accent": "#14F195"},
    "XRP":  {"name": "XRP",      "ticker": "XRP-USD",  "symbol": "XRPUSDT",  "emoji": "⚪", "accent": "#23292F"},
    "DOGE": {"name": "Dogecoin", "ticker": "DOGE-USD", "symbol": "DOGEUSDT", "emoji": "🐕", "accent": "#C2A633"},
    "ADA":  {"name": "Cardano",  "ticker": "ADA-USD",  "symbol": "ADAUSDT",  "emoji": "🔵", "accent": "#0033AD"},
    "TRX":  {"name": "TRON",     "ticker": "TRX-USD",  "symbol": "TRXUSDT",  "emoji": "🔴", "accent": "#FF060A"},
    "SUI":  {"name": "Sui",      "ticker": "SUI-USD",  "symbol": "SUIUSDT",  "emoji": "💧", "accent": "#4DA2FF"},
}


def coin_dir(coin: str) -> Path:
    return MODELS_DIR / coin


def model_path(coin: str) -> Path:
    return coin_dir(coin) / "fib_pattern_engine_v5.pkl"


def dataset_path(coin: str) -> Path:
    return coin_dir(coin) / f"Dataset Output Otomatis {coin} 2025-2026.xlsx"


def predictions_path(coin: str) -> Path:
    return coin_dir(coin) / "backtest_v5_predictions.csv"


def available_coins() -> list[str]:
    """Koin yang model .pkl-nya benar-benar ada di disk."""
    return [c for c in COINS if model_path(c).exists()]


def label(coin: str) -> str:
    c = COINS[coin]
    return f"{c['emoji']}  {coin} — {c['name']}"
