"""
Fib Path Analyzer V5 — Multi-Coin
=================================
Satu app untuk SEMUA koin. Tiap koin punya model + dataset + backtest sendiri
di models/<COIN>/ (tidak digabung, tidak diubah). Pilih koin di sidebar -> app
load .pkl koin tsb dan menjalankan engine analisa yang sama persis dengan V5.
"""
from __future__ import annotations

import html
import json
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd
import streamlit as st

from fib_pattern_engine_v5 import FibPatternEngineV5
from auto_setup import fetch_setup
from coin_registry import (
    COINS, available_coins, label, model_path, dataset_path, predictions_path,
)
from model_perf import compute_perf

# ============================ CONFIG ============================
st.set_page_config(
    page_title="Fib Path Analyzer V5 — Multi-Coin",
    layout="wide",
    page_icon="📈",
    initial_sidebar_state="expanded",
)

ACTIONABLE_TARGETS = ["1.61_UP", "1.61_DOWN", "2.5_UP", "2.5_DOWN", "3.6_UP", "3.6_DOWN"]
FIRST_HIT_TARGETS = ACTIONABLE_TARGETS + ["TIE_SAME_BAR", "NO_HIT_48H"]
CONTINUATION_ORDER = [
    "UP_1.61_TO_2.5", "UP_2.5_TO_3.6", "UP_1.61_TO_3.6",
    "DOWN_1.61_TO_2.5", "DOWN_2.5_TO_3.6", "DOWN_1.61_TO_3.6",
]
MOMENTUM_EMOJI = {"lime": "🟢", "green": "🟩", "red": "🟥", "maroon": "🟫"}
SQUEEZE_EMOJI = {"Squeeze ON (black)": "⬛", "Squeeze OFF (gray)": "⬜"}

# Sama persis dgn fib_dataset_export.FIBSPEC: Up_X = max(O,C)+(X-1)*body ; Down_X = min(O,C)-(X-1)*body
FIBSPEC = [("1.61", 0.61), ("2.5", 1.5), ("3.6", 2.6)]


def fib_price_levels(o, c):
    if o is None or c is None:
        return None
    bt, bb = max(float(o), float(c)), min(float(o), float(c))
    body = bt - bb
    if body <= 0:
        return None
    levels = {}
    for tag, mu in FIBSPEC:
        levels[f"{tag}_UP"] = bt + mu * body
        levels[f"{tag}_DOWN"] = bb - mu * body
    return {"body_top": bt, "body_bottom": bb, "body": body, "levels": levels}


def _fmt_px(v):
    if v is None:
        return "—"
    d = 2 if v >= 1000 else 3 if v >= 100 else 4 if v >= 1 else 6
    return f"{v:,.{d}f}"


# ============================ CACHING ============================
# Cache di-key oleh sidik jari file, bukan cuma nama koin. Tanpa ini app yang
# sedang berjalan tetap menyajikan angka lama setelah regenerate.py/backtest_v5.py
# menimpa .pkl dan .csv-nya.
def _stamp(path) -> tuple[int, int]:
    try:
        s = path.stat()
        return (s.st_mtime_ns, s.st_size)
    except OSError:
        return (0, 0)


@st.cache_resource(show_spinner=False, max_entries=len(COINS))
def _load_engine(coin: str, stamp: tuple) -> FibPatternEngineV5:
    return FibPatternEngineV5.load(model_path(coin))


def load_engine(coin: str) -> FibPatternEngineV5:
    return _load_engine(coin, _stamp(model_path(coin)))


@st.cache_data(show_spinner=False)
def _perf(coin: str, stamp: tuple):
    return compute_perf(predictions_path(coin))


def perf(coin: str):
    return _perf(coin, _stamp(predictions_path(coin)))


@st.cache_data(show_spinner=False)
def _all_perf_table(stamps: tuple) -> pd.DataFrame:
    rows = []
    for c in available_coins():
        p = perf(c)
        if not p:
            continue
        rows.append({
            "Koin": c, "Nama": COINS[c]["name"],
            "Arah": p["dir_acc"], "Edge vs base": p["edge"], "z": p["z"],
            "Confident-10%": p["conf_acc"], "First-hit edge": p["fh_edge"],
            "Bulan >0.5": f"{p['months_win']}/{p['months_total']}", "n": p["n_clear"],
        })
    df = pd.DataFrame(rows).sort_values("Arah", ascending=False).reset_index(drop=True)
    return df


def all_perf_table() -> pd.DataFrame:
    return _all_perf_table(tuple(_stamp(predictions_path(c)) for c in available_coins()))


# ============================ KESEGARAN MODEL ============================
# "Kapan terakhir di-update" dibaca dari mtime file, bukan disimpan manual — jadi
# otomatis benar setiap regenerate.py / backtest_v5.py menimpa file-nya.
# Semua jam di panel ini ditampilkan dalam WIB (UTC+7) — zona acuan app (caption form
# "WIB = UTC+7") — bukan zona jam mesin, supaya jam model vs jam data langsung sebanding.
# (Jam mesin ini UTC+8, jadi angka di Explorer terlihat 1 jam lebih besar — itu wajar.)
WIB = timezone(timedelta(hours=7))
TZ_LABEL = "WIB"
# Status kesegaran: glyph + kata, bukan warna saja.
_ST = {"good": ("✓", ""), "warning": ("▲", " · mulai usang"),
       "serious": ("✕", " · usang"), "muted": ("–", "")}
_MONTHS_ID_INV = {"Januari": 1, "Februari": 2, "Maret": 3, "April": 4, "Mei": 5, "Juni": 6,
                  "Juli": 7, "Agustus": 8, "September": 9, "Oktober": 10, "November": 11,
                  "Desember": 12}


def _mtime_local(path):
    try:
        return datetime.fromtimestamp(path.stat().st_mtime, tz=WIB)
    except OSError:
        return None


@st.cache_data(show_spinner=False)
def _dataset_last_bar(coin: str, stamp: tuple):
    """Bar 1h terakhir di sheet terakhir dataset (UTC). Sheet = 1 bulan, urut kronologis."""
    try:
        df = pd.read_excel(dataset_path(coin), sheet_name=-1, usecols=["Date", "Clock"])
        d, mon, y = str(df["Date"].iloc[-1]).split()
        return datetime(int(y), _MONTHS_ID_INV[mon], int(d), int(df["Clock"].iloc[-1]),
                        tzinfo=timezone.utc)
    except Exception:
        return None


def freshness(coin: str) -> dict:
    return {
        "model": _mtime_local(model_path(coin)),
        "backtest": _mtime_local(predictions_path(coin)),
        "dataset": _mtime_local(dataset_path(coin)),
        "last_bar": _dataset_last_bar(coin, _stamp(dataset_path(coin))),
    }


def _age(dt) -> tuple[str, str]:
    """('hari ini' | 'kemarin' | 'N hari lalu', status good|warning|serious|muted)."""
    if dt is None:
        return "—", "muted"
    days = (datetime.now(WIB).date() - dt.astimezone(WIB).date()).days
    txt = "hari ini" if days == 0 else "kemarin" if days == 1 else f"{days} hari lalu"
    return txt, ("good" if days <= 3 else "warning" if days <= 10 else "serious")


def _fmt_dt(dt, tz_label: str = TZ_LABEL) -> str:
    return f"{dt.astimezone(WIB):%d %b %Y %H:%M} {tz_label}" if dt else "—"


def render_freshness(fr: dict) -> str:
    """Panel kecil 3 baris: model · backtest · data. Warna status selalu ditemani teks."""
    rows = []
    for icon, lbl, dt in (("🧠", "Model diperbarui", fr["model"]),
                          ("🧪", "Backtest dijalankan", fr["backtest"]),
                          ("📡", "Data Binance s/d", fr["last_bar"])):
        age, status = _age(dt)
        glyph, word = _ST[status]
        rows.append(f"<div class='fr-row'><span class='fr-ic'>{icon}</span>"
                    f"<span class='fr-lbl'>{lbl}</span>"
                    f"<span class='fr-val'>{_fmt_dt(dt)}</span>"
                    f"<span class='fr-age st-{status}'><i>{glyph}</i>{age}{word}</span></div>")
    return ("<div class='fresh'><div class='fr-title'>🕒 Pembaruan terakhir · WIB (UTC+7)</div>"
            + "".join(rows) + "</div>")


# ============================ LADDER LEVEL FIB ============================
UP_TARGETS = ["3.6_UP", "2.5_UP", "1.61_UP"]
DOWN_TARGETS = ["1.61_DOWN", "2.5_DOWN", "3.6_DOWN"]


def render_fib_ladder(fib: dict, fib_px: dict, close_px, reach: dict, first_hit: dict) -> str:
    """Tabel level Fib sebagai HTML: Prob Reach = meter per baris (kolom utama).

    Warna bar = arah (UP biru / DOWN merah, anchor abu-abu netral); angka tetap
    warna teks. Baris reach tertinggi ditandai ikon + ring, bukan warna saja.
    Dibangun satu baris tanpa indentasi supaya markdown Streamlit tidak
    menganggapnya code block.
    """
    tg = UP_TARGETS + DOWN_TARGETS
    r = {t: float(reach.get(t, 0) or 0) for t in tg}
    fh = {t: float(first_hit.get(t, 0) or 0) for t in tg}
    r_max, fh_max = max(r.values()), max(fh.values())
    # Tandai "tertinggi" hanya bila unik: reach sering seri di 100% untuk beberapa level
    # sekaligus, dan ring/chip di separuh tabel tidak menuntun mata ke mana pun.
    eps = 1e-9
    r_tops = [t for t in tg if abs(r[t] - r_max) < eps]
    fh_tops = [t for t in tg if abs(fh[t] - fh_max) < eps]
    r_top = r_tops[0] if (len(r_tops) == 1 and r_max > 0) else None
    fh_top_t = fh_tops[0] if (len(fh_tops) == 1 and fh_max > 0) else None

    def row(t):
        side = "up" if t.endswith("_UP") else "down"
        px = fib_px.get(t)
        dist = f"{(px - close_px) / close_px:+.2%}" if (px is not None and close_px) else "—"
        top = t == r_top
        fh_top = t == fh_top_t
        pct = max(0.0, min(1.0, r[t])) * 100
        chip = "<span class='chip'>🎯 tertinggi</span>" if top else ""
        fh_cls = "fh dim" if fh[t] == 0 else ("fh strong" if fh_top else "fh")
        fh_chip = "<span class='chip chip-fh'>1st</span>" if fh_top else ""
        return (f"<div class='lr {side}{' top' if top else ''}'>"
                f"<div class='c tgt'><i class='dot'></i><span class='tl'>{html.escape(t)}</span>{chip}</div>"
                f"<div class='c px'>{_fmt_px(px)}</div>"
                f"<div class='c dist'>{dist}</div>"
                f"<div class='c reach' title='Prob Reach {html.escape(t)}: {r[t]:.1%}'>"
                f"<span class='rv'>{r[t]:.1%}</span>"
                f"<span class='track'><span class='fill' style='width:{pct:.1f}%'></span></span></div>"
                f"<div class='c {fh_cls}'>{fh[t]:.1%}{fh_chip}</div>"
                f"</div>")

    anchor = (f"<div class='lr anchor'><div class='c tgt'><i class='dot'></i><span class='tl'>close anchor</span></div>"
              f"<div class='c px'>{_fmt_px(close_px)}</div><div class='c dist'>0.00%</div>"
              f"<div class='c reach'><span class='mid'></span></div><div class='c fh dim'></div></div>")

    head = ("<div class='lh'><div>Target</div><div>Harga</div><div>Jarak dari close</div>"
            "<div>Prob Reach</div><div>Prob First-hit</div></div>")
    title = (f"<div class='lt'><div><span class='lt-h'>💰 Level Harga Fib</span>"
             f"<span class='lt-s'>body anchor {_fmt_px(fib['body_bottom'])} – {_fmt_px(fib['body_top'])}"
             f" · body {_fmt_px(fib['body'])}</span></div>"
             f"<div class='lg'><span class='lg-up'><i></i>UP</span><span class='lg-dn'><i></i>DOWN</span>"
             f"<span class='lg-note'>bar = Prob Reach (0–100%)</span></div></div>")
    body = "".join(row(t) for t in UP_TARGETS) + anchor + "".join(row(t) for t in DOWN_TARGETS)
    return f"<div class='ladder'>{title}{head}{body}</div>"


def inject_css(accent: str) -> None:
    st.markdown(f"""
    <style>
      .stApp {{ background: linear-gradient(180deg,#0e1117 0%, #0e1117 60%, #111722 100%); }}
      .coin-hero {{
          padding: 18px 22px; border-radius: 16px; margin-bottom: 8px;
          background: radial-gradient(120% 140% at 0% 0%, {accent}26 0%, #1b2230 55%, #161b26 100%);
          border: 1px solid {accent}55; box-shadow: 0 6px 24px #0006;
      }}
      .coin-hero h1 {{ margin:0; font-size: 2.0rem; letter-spacing:.5px; }}
      .coin-hero .sub {{ color:#aab3c5; font-size:.92rem; margin-top:2px; }}
      .pill {{ display:inline-block; padding:3px 10px; border-radius:999px; font-size:.78rem;
               background:{accent}22; border:1px solid {accent}66; color:#dfe6f2; margin-right:6px; }}
      div[data-testid="stMetric"] {{
          background:#161b26; border:1px solid #232a39; border-radius:12px;
          padding:12px 14px;
      }}
      div[data-testid="stMetricValue"] {{ font-size:1.5rem; }}
      .stTabs [data-baseweb="tab-list"] {{ gap: 4px; }}
      .stTabs [data-baseweb="tab"] {{ background:#161b26; border-radius:10px 10px 0 0; padding:8px 16px; }}
      .stTabs [aria-selected="true"] {{ background:{accent}26; border-bottom:2px solid {accent}; }}
      .stApp {{ --accent:{accent}; --accent-border:{accent}55; }}
      {LADDER_CSS}
    </style>
    """, unsafe_allow_html=True)


# CSS ladder + panel kesegaran. Token warna: surface app #161b26 / #232a39; tinta
# #eef2f8 / #aab3c5 / #6f7a90; seri UP #3987e5, DOWN #e66767 (pasangan divergen
# biru↔merah, lolos cek CVD di surface gelap — hijau/merah tidak); status good
# #0ca30c, warning #fab219, serious #ec835a.
LADDER_CSS = """
      .ladder { background:#161b26; border:1px solid var(--accent-border); border-radius:16px;
                padding:14px 18px 10px; margin:4px 0 14px; box-shadow:0 6px 24px #0006;
                font-family:system-ui,-apple-system,"Segoe UI",sans-serif;
                container-type:inline-size; }
      .ladder .lt { display:flex; justify-content:space-between; align-items:flex-end; gap:12px;
                    flex-wrap:wrap; margin-bottom:10px; }
      .ladder .lt-h { font-size:1.12rem; font-weight:700; color:#eef2f8; margin-right:10px; }
      .ladder .lt-s { font-size:.84rem; color:#aab3c5; }
      .ladder .lg { display:flex; gap:14px; align-items:center; font-size:.78rem; color:#aab3c5; }
      .ladder .lg i { display:inline-block; width:10px; height:10px; border-radius:2px; margin-right:5px;
                      vertical-align:-1px; }
      .ladder .lg-up i { background:#3987e5; }  .ladder .lg-dn i { background:#e66767; }
      .ladder .lg-note { color:#6f7a90; }
      .ladder .lh, .ladder .lr { display:grid; align-items:center; gap:0 14px;
        grid-template-columns:minmax(130px,1.4fr) minmax(100px,1.1fr) minmax(96px,1fr) minmax(200px,3fr) minmax(96px,1fr); }
      .ladder .lh { font-size:.74rem; letter-spacing:.06em; text-transform:uppercase; color:#6f7a90;
                    padding:6px 10px; border-bottom:1px solid #232a39; }
      .ladder .lh div:nth-child(4) { color:#aab3c5; font-weight:700; }
      .ladder .lr { padding:9px 10px; min-height:50px; border-radius:10px; border:1px solid transparent;
                    transition:background .12s; font-variant-numeric:tabular-nums; }
      .ladder .lr + .lr { margin-top:2px; }
      .ladder .lr:hover { background:#ffffff08; }
      .ladder .c { min-width:0; color:#eef2f8; font-size:.98rem; }
      .ladder .tgt { font-weight:600; display:flex; align-items:center; gap:6px 8px; flex-wrap:wrap; }
      .ladder .tl { white-space:nowrap; }
      .ladder .dot { width:8px; height:8px; border-radius:50%; flex:0 0 8px; }
      .ladder .up .dot { background:#3987e5; }  .ladder .down .dot { background:#e66767; }
      .ladder .dist { color:#aab3c5; }
      .ladder .reach { display:flex; align-items:center; gap:12px; }
      .ladder .rv { font-size:1.3rem; font-weight:700; color:#eef2f8; flex:0 0 5.4rem; min-width:5.4rem;
                    text-align:right; line-height:1; }
      .ladder .track { flex:1 1 auto; height:10px; border-radius:0 4px 4px 0; overflow:hidden; background:#2a3140; }
      .ladder .fill { display:block; height:100%; border-radius:0 4px 4px 0; }
      .ladder .up .track { background:#3987e52e; }   .ladder .up .fill { background:#3987e5; }
      .ladder .down .track { background:#e666672e; } .ladder .down .fill { background:#e66767; }
      .ladder .fh { color:#aab3c5; }
      .ladder .fh.strong { color:#eef2f8; font-weight:700; }
      .ladder .fh.dim { color:#6f7a90; }
      .ladder .chip { font-size:.7rem; font-weight:600; letter-spacing:.02em; color:#eef2f8;
                      background:#ffffff14; border:1px solid #ffffff22; border-radius:999px;
                      padding:2px 8px; white-space:nowrap; }
      .ladder .chip-fh { margin-left:8px; color:#aab3c5; }
      .ladder .top { background:#ffffff0a; border-color:#ffffff24; }
      .ladder .top .rv { font-size:1.45rem; }
      .ladder .anchor { background:#2a314066; margin:6px 0; min-height:38px; padding:5px 10px; }
      .ladder .anchor .c { color:#aab3c5; font-size:.9rem; }
      .ladder .anchor .tgt { font-weight:600; letter-spacing:.04em; text-transform:uppercase; font-size:.78rem; }
      .ladder .anchor .dot { background:#6f7a90; }
      .ladder .anchor .mid { flex:1; height:1px; background:#3a4356; }
      /* Lebar sempit (sidebar terbuka di laptop kecil / jendela dibagi dua): tiap baris jadi
         2 lajur — target·harga·jarak di atas, meter selebar kartu + first-hit di bawah. */
      @container (max-width: 760px) {
        .ladder .lh { display:none; }
        .ladder .lr { grid-template-columns:minmax(0,1fr) auto 12rem; grid-template-areas:"tgt px dist" "reach reach fh";
                      row-gap:8px; padding:10px 12px; }
        .ladder .tgt { grid-area:tgt; } .ladder .px { grid-area:px; }
        .ladder .dist { grid-area:dist; text-align:right; }
        .ladder .reach { grid-area:reach; } .ladder .fh { grid-area:fh; text-align:right; white-space:nowrap; }
        .ladder .fh::before { content:"first-hit "; color:#6f7a90; font-size:.72rem; letter-spacing:.05em;
                              text-transform:uppercase; }
        .ladder .anchor { grid-template-areas:"tgt px dist"; }
        .ladder .anchor .reach, .ladder .anchor .fh { display:none; }
      }
      .fresh { background:#161b26; border:1px solid #232a39; border-radius:12px; padding:10px 12px; margin:6px 0 2px;
               font-family:system-ui,-apple-system,"Segoe UI",sans-serif; }
      .fresh .fr-title { font-size:.78rem; letter-spacing:.05em; text-transform:uppercase; color:#6f7a90;
                         margin-bottom:6px; }
      .fresh .fr-row { display:grid; grid-template-columns:18px 1fr; gap:0 6px; align-items:baseline;
                       padding:3px 0; font-size:.82rem; }
      .fresh .fr-ic { font-size:.8rem; }
      .fresh .fr-lbl { color:#aab3c5; }
      .fresh .fr-val { grid-column:2; color:#eef2f8; font-variant-numeric:tabular-nums; font-weight:600; }
      .fresh .fr-age { grid-column:2; font-size:.76rem; color:#aab3c5; }
      .fresh .fr-age i { font-style:normal; font-weight:700; margin-right:5px; color:#6f7a90; }
      .fresh .st-good i { color:#0ca30c; }  .fresh .st-warning i { color:#fab219; }
      .fresh .st-serious i { color:#ec835a; }
      .hero-fresh { display:inline-flex; align-items:center; gap:6px; margin-left:10px; padding:2px 10px;
                    border-radius:999px; background:#ffffff10; border:1px solid #ffffff1f; font-size:.8rem;
                    color:#dfe6f2; }
      .hero-fresh i { font-style:normal; font-weight:700; color:#6f7a90; }
      .hero-fresh.st-good i { color:#0ca30c; } .hero-fresh.st-warning i { color:#fab219; }
      .hero-fresh.st-serious i { color:#ec835a; }
"""


# ============================ SIDEBAR ============================
coins = available_coins()
if not coins:
    st.error("Tidak ada model di folder `models/`. Pastikan models/<COIN>/fib_pattern_engine_v5.pkl ada.")
    st.stop()

with st.sidebar:
    st.markdown("## 📈 Fib Path Analyzer V5")
    st.caption("Multi-Coin · engine k-NN + exact-match · data Binance 1h")

    coin = st.selectbox("Pilih koin", coins, format_func=label, key="coin")
    meta = COINS[coin]

    st.markdown(
        f"<div class='pill'>{meta['emoji']} {coin}</div>"
        f"<div class='pill'>{meta['symbol']}</div>", unsafe_allow_html=True)

    p = perf(coin)
    if p:
        st.metric("🎯 Akurasi arah (OOS)", f"{p['dir_acc']:.1%}",
                  f"{p['edge']:+.1%} vs base-rate")
        st.caption(f"Konsisten **{p['months_win']}/{p['months_total']}** bulan · z={p['z']:.1f} "
                   f"· prediksi s/d **{p['date_max']:%d %b %Y}**")

    fresh = freshness(coin)
    st.markdown(render_freshness(fresh), unsafe_allow_html=True)

    ds = dataset_path(coin)
    st.markdown("---")
    st.caption(f"Model: `models/{coin}/fib_pattern_engine_v5.pkl`")
    st.caption(f"Dataset: `{ds.name}` {'✅' if ds.exists() else '❌'}")
    st.caption(f"Total model tersedia: **{len(coins)}** koin")


# ============================ HERO HEADER ============================
inject_css(meta["accent"])
engine = load_engine(coin)

_age_txt, _age_st = _age(fresh["model"])
st.markdown(f"""
<div class="coin-hero">
  <h1>{meta['emoji']} {coin}/USD — {meta['name']}</h1>
  <div class="sub">Fib Path Analyzer V5 · {meta['symbol']} · 1h Binance
    <span class="hero-fresh st-{_age_st}"><i>{_ST[_age_st][0]}</i>model diperbarui {_fmt_dt(fresh['model'])} · {_age_txt}{_ST[_age_st][1]}</span>
  </div>
</div>
""", unsafe_allow_html=True)

tab_analisa, tab_perf, tab_about = st.tabs(["🎯 Analisa Setup", "📈 Performa Model", "📊 Semua Model"])


# ============================ TAB 1: ANALISA ============================
with tab_analisa:
    now_utc = datetime.now(timezone.utc)
    with st.form("input_form"):
        c1, c2, c3, c4 = st.columns([2, 2, 2, 1.6])
        with c1:
            st.text_input("Ticker", value=meta["ticker"], disabled=True,
                          help="Otomatis dari koin terpilih. Data di-fetch dari Binance.")
        with c2:
            date_val = st.date_input("Tanggal (UTC)", value=now_utc.date(), max_value=now_utc.date())
        with c3:
            hour_val = st.slider("Jam (UTC)", 0, 23, now_utc.hour)
        with c4:
            trend_val = st.radio("Trend", ["Long", "Short"], horizontal=True)
        st.caption(f"🕐 Sekarang **{now_utc:%Y-%m-%d %H:%M} UTC** (WIB = UTC+7). "
                   f"Pilih jam ≤ {now_utc.hour:02d}:00 untuk hari ini.")
        submitted = st.form_submit_button("🔮 Jalankan Prediksi", use_container_width=True, type="primary")

    if submitted:
        target_dt = datetime(date_val.year, date_val.month, date_val.day,
                             int(hour_val), tzinfo=timezone.utc)
        if target_dt > datetime.now(timezone.utc):
            st.error(f"⏳ {date_val} {hour_val:02d}:00 UTC **masih di masa depan** — sekarang baru "
                     f"{datetime.now(timezone.utc):%H:%M} UTC ({date_val} {hour_val:02d}:00 UTC = "
                     f"jam {(int(hour_val) + 7) % 24:02d}:00 WIB). Pilih jam yang sudah lewat.")
            st.stop()
        with st.spinner(f"📡 Fetch setup {coin} dari Binance..."):
            setup_auto = fetch_setup(meta["ticker"], date_val, hour_val)
        if setup_auto.get("error"):
            st.error(f"⚠️ Auto-fetch gagal: {setup_auto['error']}")
            st.stop()
        setup_data = {
            "Trend": trend_val,
            "SQZMOM 1 Momentum": setup_auto["SQZMOM 1 Momentum"],
            "SQZMOM 1 Squeeze": setup_auto["SQZMOM 1 Squeeze"],
            "SQZMOM 1 Value": setup_auto["SQZMOM 1 Value"],
            "SQZMOM 2 Momentum": setup_auto["SQZMOM 2 Momentum"],
            "SQZMOM 2 Squeeze": setup_auto["SQZMOM 2 Squeeze"],
            "SQZMOM 2 Value": setup_auto["SQZMOM 2 Value"],
            "Bar 1": setup_auto["Bar 1"], "Bar 2": setup_auto["Bar 2"],
            "Raw Position": setup_auto["Raw Position"],
            "Final Position": setup_auto["Final Position"],
            "Score": setup_auto["Score"], "Last TR": setup_auto["Last TR"],
        }
        with st.spinner("🧠 Menganalisis pola historis..."):
            result = engine.predict(setup_data, top_k_matches=5)
        # Simpan ke session_state supaya hasil TETAP tampil saat tombol download
        # di-klik (download memicu rerun, submitted kembali False).
        st.session_state["last_run"] = {
            "coin": coin, "date": str(date_val), "hour": int(hour_val),
            "trend": trend_val, "setup_auto": setup_auto, "result": result,
        }

    _last = st.session_state.get("last_run")
    if _last and _last.get("coin") == coin:
        setup_auto = _last["setup_auto"]
        result = _last["result"]
        run_date, run_hour, run_trend = _last["date"], _last["hour"], _last["trend"]

        close_val = setup_auto.get("_close")
        st.divider()
        st.subheader(f"🤖 Setup {coin} di {run_date} jam {run_hour:02d}:00 UTC")
        if close_val is not None:
            st.caption(f"Close price: **{close_val:,.4f}**")

        bcol, s1col, s2col, sigcol = st.columns([1.1, 1.3, 1.3, 1.5])
        with bcol:
            st.markdown("**🕯️ Bar**")
            st.metric("Bar 1 (jam ini)", setup_auto["Bar 1"])
            st.metric("Bar 2 (jam lalu)", setup_auto["Bar 2"])
        with s1col:
            m, sq = setup_auto["SQZMOM 1 Momentum"], setup_auto["SQZMOM 1 Squeeze"]
            st.markdown(f"**📊 SQZMOM 1** {MOMENTUM_EMOJI.get(m,'')} {SQUEEZE_EMOJI.get(sq,'')}")
            st.metric("Value", f"{setup_auto['SQZMOM 1 Value']:+.4f}")
            st.caption(f"Momentum: `{m}` · Squeeze: `{sq}`")
        with s2col:
            m, sq = setup_auto["SQZMOM 2 Momentum"], setup_auto["SQZMOM 2 Squeeze"]
            st.markdown(f"**📊 SQZMOM 2** {MOMENTUM_EMOJI.get(m,'')} {SQUEEZE_EMOJI.get(sq,'')}")
            st.metric("Value", f"{setup_auto['SQZMOM 2 Value']:+.4f}")
            st.caption(f"Momentum: `{m}` · Squeeze: `{sq}`")
        with sigcol:
            st.markdown("**⚡ Signals**")
            a, b = st.columns(2)
            a.metric("Score", f"{setup_auto['Score']}")
            b.metric("Last TR", f"{setup_auto['Last TR']:.4f}")
            a.metric("Raw Position", setup_auto["Raw Position"])
            b.metric("Final Position", setup_auto["Final Position"])

        with st.expander("📋 Detail Indikator Market"):
            d1, d2, d3, d4 = st.columns(4)
            d1.metric("Last Close", f"{setup_auto.get('last_close') or 0:.4f}")
            d2.metric("RSI 14", f"{setup_auto.get('rsi_last') or 0:.2f}")
            d3.metric("ADX 14", f"{setup_auto.get('adx_last') or 0:.2f}")
            d4.metric("ATR 14", f"{setup_auto.get('atr_last') or 0:.4f}")
            d1.metric("EMA 21", f"{setup_auto.get('ema_fast_last') or 0:.4f}")
            d2.metric("EMA 50", f"{setup_auto.get('ema_slow_last') or 0:.4f}")
            d3.metric("MACD", f"{setup_auto.get('macd_last') or 0:.4f}")
            d4.metric("Filter", setup_auto.get("filter_reason", "-"))

        st.divider()
        st.header("📊 Hasil Prediksi")

        fib = fib_price_levels(setup_auto.get("_open"), setup_auto.get("_close"))
        fib_px = fib["levels"] if fib else {}

        def _with_px(target):
            px = fib_px.get(target)
            return f"{target} @ {_fmt_px(px)}" if px is not None else (target or "-")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("1️⃣ First-hit utama", _with_px(result.first_hit_top_target), f"{result.first_hit_top_prob:.1%}")
        m2.metric("2️⃣ Kemungkinan kedua", _with_px(result.first_hit_second_target), f"{result.first_hit_second_prob:.1%}")
        m3.metric("⚠️ Risk Tie Same Bar", f"{result.tie_prob:.1%}")
        m4.metric("⌚ Risk No Hit 48h", f"{result.no_hit_prob:.1%}")

        reach_sorted = sorted(result.reach_probs.items(), key=lambda x: x[1], reverse=True)
        r1, r2 = st.columns(2)
        tr, trp = (reach_sorted[0] if reach_sorted else ("-", 0.0))
        sr, srp = (reach_sorted[1] if len(reach_sorted) > 1 else ("-", 0.0))
        r1.metric("🎯 Reach paling mungkin", _with_px(tr), f"{trp:.1%}")
        r2.metric("🎯 Reach kedua", _with_px(sr), f"{srp:.1%}")

        if fib:
            close_px = setup_auto.get("_close")
            st.markdown(render_fib_ladder(fib, fib_px, close_px, result.reach_probs,
                                          result.first_hit_probs), unsafe_allow_html=True)

            # --- Export untuk database ---
            dt_utc = f"{run_date} {run_hour:02d}:00"
            wide = {
                "coin": coin, "datetime_utc": dt_utc, "trend": run_trend,
                "anchor_open": setup_auto.get("_open"), "anchor_high": setup_auto.get("_high"),
                "anchor_low": setup_auto.get("_low"), "anchor_close": close_px, "body": fib["body"],
                "first_hit_top": result.first_hit_top_target,
                "first_hit_top_prob": result.first_hit_top_prob,
                "tie_prob": result.tie_prob, "no_hit_prob": result.no_hit_prob,
            }
            for t in ACTIONABLE_TARGETS:
                col = t.replace(".", "").lower()  # 1.61_UP -> 161_up
                wide[f"px_{col}"] = fib_px.get(t)
                wide[f"reach_{col}"] = result.reach_probs.get(t)
                wide[f"fh_{col}"] = result.first_hit_probs.get(t)
            csv_str = pd.DataFrame([wide]).to_csv(index=False)
            payload = {
                "coin": coin, "datetime_utc": dt_utc, "trend": run_trend,
                "anchor": {"open": setup_auto.get("_open"), "high": setup_auto.get("_high"),
                           "low": setup_auto.get("_low"), "close": close_px, "body": fib["body"]},
                "fib_prices": fib_px,
                "reach_probs": result.reach_probs,
                "first_hit_probs": result.first_hit_probs,
                "continuation_probs": result.continuation_probs,
            }
            json_str = json.dumps(payload, indent=2, default=str)
            fname = f"fibpath_{coin}_{run_date}_{run_hour:02d}00UTC"
            e1, e2 = st.columns(2)
            e1.download_button("⬇️ CSV (1 baris — siap database)", csv_str,
                               f"{fname}.csv", "text/csv", use_container_width=True)
            e2.download_button("⬇️ JSON (lengkap)", json_str,
                               f"{fname}.json", "application/json", use_container_width=True)

        g1, g2 = st.columns(2)
        with g1:
            st.markdown("##### Distribusi First-hit")
            st.bar_chart(pd.DataFrame({
                "Target": FIRST_HIT_TARGETS,
                "Prob (%)": [result.first_hit_probs.get(k, 0.0) * 100 for k in FIRST_HIT_TARGETS],
            }).set_index("Target"))
        with g2:
            st.markdown("##### Reach Probability Semua Fib")
            st.bar_chart(pd.DataFrame({
                "Target": ACTIONABLE_TARGETS,
                "Prob (%)": [result.reach_probs.get(k, 0.0) * 100 for k in ACTIONABLE_TARGETS],
            }).set_index("Target"))

        g3, g4 = st.columns([2, 1])
        with g3:
            st.markdown("##### Continuation Probability")
            st.bar_chart(pd.DataFrame({
                "Transition": CONTINUATION_ORDER,
                "Prob (%)": [result.continuation_probs.get(k, 0.0) * 100 for k in CONTINUATION_ORDER],
            }).set_index("Transition"))
        with g4:
            st.markdown("##### Sumber Keputusan")
            st.metric("Exact Match", f"{result.source_summary.get('exact_match_count', 0):.0f} data")
            st.metric("Bobot Exact", f"{result.source_summary.get('exact_weight_used', 0.0):.1%}")
            st.metric("Bobot Similarity", f"{result.source_summary.get('similarity_weight_used', 0.0):.1%}")

        st.subheader("📚 Top Kasus Historis Paling Mirip")
        if result.top_matches:
            mdf = pd.DataFrame(result.top_matches).rename(columns={
                "date": "Tanggal", "clock": "Jam", "first_hit_target": "First Hit",
                "first_hit_direction": "Arah", "first_hit_level": "Level",
                "reached_targets": "Fib Tercapai", "similarity": "Kemiripan",
                "trend": "Trend", "score": "Score", "last_tr": "Last TR",
                "raw_position": "Raw Position", "final_position": "Final Position",
            })
            if "Tanggal" in mdf:
                mdf["Tanggal"] = mdf["Tanggal"].astype(str).replace("NaT", "-")
            if "Kemiripan" in mdf:
                mdf["Kemiripan"] = mdf["Kemiripan"].apply(lambda x: f"{x:.1%}")
            cols = [c for c in ["Tanggal", "Jam", "First Hit", "Arah", "Level", "Fib Tercapai",
                                "Kemiripan", "Trend", "Score", "Last TR", "Raw Position", "Final Position"]
                    if c in mdf.columns]
            st.dataframe(mdf[cols], use_container_width=True, hide_index=True)
        else:
            st.info("Tidak ada kasus historis yang cocok.")
    else:
        st.caption("Isi Tanggal · Jam · Trend → Jalankan Prediksi.")


# ============================ TAB 2: PERFORMA ============================
with tab_perf:
    p = perf(coin)
    if not p:
        st.warning("Belum ada backtest_v5_predictions.csv untuk koin ini.")
    else:
        _bt = freshness(coin)["backtest"]
        st.markdown(f"#### {coin} · {p['n_total']:,} prediksi · "
                    f"{pd.to_datetime(p['date_min']):%b %Y}–{pd.to_datetime(p['date_max']):%b %Y}")
        st.caption(f"🧪 Backtest terakhir dijalankan **{_fmt_dt(_bt, TZ_LABEL)}** "
                   f"({_age(_bt)[0]}) · file `models/{coin}/backtest_v5_predictions.csv`")
        k1, k2, k3, k4, k5, k6 = st.columns(6)
        k1.metric("Akurasi arah", f"{p['dir_acc']:.1%}", f"{p['edge']:+.1%} vs base")
        k2.metric("Signifikansi z", f"{p['z']:.1f}")
        k3.metric("Confident-10%", f"{p['conf_acc']:.1%}", f"{p['conf_acc']-p['dir_acc']:+.1%}")
        k4.metric("Bulan unggul", f"{p['months_win']}/{p['months_total']}")
        k5.metric("vs ikut-tren", f"{p['edge_vs_naive']:+.1%}")
        k6.metric("First-hit edge", f"{p['fh_edge']:+.1%}")

        cA, cB = st.columns(2)
        with cA:
            st.markdown("##### Akurasi arah per bulan")
            st.bar_chart(p["per_month"].set_index("month")["acc"])
        with cB:
            st.markdown("##### Akurasi vs top-X% confidence")
            cc = p["conf_curve"].copy()
            cc["label"] = (cc["top_frac"] * 100).map(lambda v: f"top {v:.0f}%")
            st.bar_chart(cc.set_index("label")["acc"])


# ============================ TAB 3: TENTANG ============================
with tab_about:
    tbl = all_perf_table()
    st.dataframe(
        tbl.style.format({
            "Arah": "{:.1%}", "Edge vs base": "{:+.1%}", "z": "{:.1f}",
            "Confident-10%": "{:.1%}", "First-hit edge": "{:+.1%}", "n": "{:,}",
        }).background_gradient(subset=["Arah", "z"], cmap="Greens"),
        use_container_width=True, hide_index=True,
    )
