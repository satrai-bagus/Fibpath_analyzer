"""
Fib Path Analyzer V5 — Multi-Coin
=================================
Satu app untuk SEMUA koin. Tiap koin punya model + dataset + backtest sendiri
di models/<COIN>/ (tidak digabung, tidak diubah). Pilih koin di sidebar -> app
load .pkl koin tsb dan menjalankan engine analisa yang sama persis dengan V5.
"""
from __future__ import annotations

from datetime import date

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


# ============================ CACHING ============================
@st.cache_resource(show_spinner=False)
def load_engine(coin: str) -> FibPatternEngineV5:
    return FibPatternEngineV5.load(model_path(coin))


@st.cache_data(show_spinner=False)
def perf(coin: str):
    return compute_perf(predictions_path(coin))


@st.cache_data(show_spinner=False)
def all_perf_table() -> pd.DataFrame:
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
    </style>
    """, unsafe_allow_html=True)


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
        st.caption(f"Konsisten **{p['months_win']}/{p['months_total']}** bulan · z={p['z']:.1f}")

    ds = dataset_path(coin)
    st.markdown("---")
    st.caption(f"Model: `models/{coin}/fib_pattern_engine_v5.pkl`")
    st.caption(f"Dataset: `{ds.name}` {'✅' if ds.exists() else '❌'}")
    st.caption(f"Total model tersedia: **{len(coins)}** koin")


# ============================ HERO HEADER ============================
inject_css(meta["accent"])
engine = load_engine(coin)

st.markdown(f"""
<div class="coin-hero">
  <h1>{meta['emoji']} {coin}/USD — {meta['name']}</h1>
  <div class="sub">Fib Path Analyzer V5 · {meta['symbol']} · 1h Binance</div>
</div>
""", unsafe_allow_html=True)

tab_analisa, tab_perf, tab_about = st.tabs(["🎯 Analisa Setup", "📈 Performa Model", "📊 Semua Model"])


# ============================ TAB 1: ANALISA ============================
with tab_analisa:
    with st.form("input_form"):
        c1, c2, c3, c4 = st.columns([2, 2, 2, 1.6])
        with c1:
            st.text_input("Ticker", value=meta["ticker"], disabled=True,
                          help="Otomatis dari koin terpilih. Data di-fetch dari Binance.")
        with c2:
            date_val = st.date_input("Tanggal (UTC)", value=date.today())
        with c3:
            hour_val = st.slider("Jam (UTC)", 0, 23, 0)
        with c4:
            trend_val = st.radio("Trend", ["Long", "Short"], horizontal=True)
        submitted = st.form_submit_button("🔮 Jalankan Prediksi", use_container_width=True, type="primary")

    if submitted:
        with st.spinner(f"📡 Fetch setup {coin} dari Binance..."):
            setup_auto = fetch_setup(meta["ticker"], date_val, hour_val)
        if setup_auto.get("error"):
            st.error(f"⚠️ Auto-fetch gagal: {setup_auto['error']}")
            st.stop()

        close_val = setup_auto.get("_close")
        st.divider()
        head = f"🤖 Setup {coin} di {date_val} jam {hour_val:02d}:00 UTC"
        st.subheader(head)
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

        st.divider()
        st.header("📊 Hasil Prediksi")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("1️⃣ First-hit utama", result.first_hit_top_target or "-", f"{result.first_hit_top_prob:.1%}")
        m2.metric("2️⃣ Kemungkinan kedua", result.first_hit_second_target or "-", f"{result.first_hit_second_prob:.1%}")
        m3.metric("⚠️ Risk Tie Same Bar", f"{result.tie_prob:.1%}")
        m4.metric("⌚ Risk No Hit 48h", f"{result.no_hit_prob:.1%}")

        reach_sorted = sorted(result.reach_probs.items(), key=lambda x: x[1], reverse=True)
        r1, r2 = st.columns(2)
        tr, trp = (reach_sorted[0] if reach_sorted else ("-", 0.0))
        sr, srp = (reach_sorted[1] if len(reach_sorted) > 1 else ("-", 0.0))
        r1.metric("🎯 Reach paling mungkin", tr, f"{trp:.1%}")
        r2.metric("🎯 Reach kedua", sr, f"{srp:.1%}")

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
        st.markdown(f"#### {coin} · {p['n_total']:,} prediksi · "
                    f"{pd.to_datetime(p['date_min']):%b %Y}–{pd.to_datetime(p['date_max']):%b %Y}")
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
