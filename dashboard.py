import streamlit as st
import pickle
import gdown
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Trading Sentiment Predictor",
    page_icon="📈",
    layout="wide"
)

# ── Load Model from Google Drive ───────────────────────────────────────────────
@st.cache_resource
def load_model():
    model_path = "model.pkl"
    if not os.path.exists(model_path):
        with st.spinner("Downloading model... please wait ⏳"):
            url = "https://drive.google.com/uc?id=1mcUPUXruvJ9DIABFcw1c41lqR8kuAGiv"
            gdown.download(url, model_path, quiet=False)
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

SENTIMENT_ORDER = ["Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"]
SENT_MAP        = {"Extreme Fear": 1, "Fear": 2, "Neutral": 3, "Greed": 4, "Extreme Greed": 5}

# ── Header ─────────────────────────────────────────────────────────────────────
st.title("📈 Next-Day Profitability Predictor")
st.markdown("Enter **today's** trading conditions to predict **tomorrow's** likely PnL outcome.")
st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# INPUT SECTION — 4 groups matching exact feature_cols from notebook
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("📥 Input Features")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("**📊 Today's Sentiment & Activity**")
    inp_sent      = st.selectbox("Market Sentiment", SENTIMENT_ORDER, index=2)
    inp_tc        = st.number_input("Trade Count", min_value=1, max_value=500, value=10)
    inp_wr        = st.slider("Win Rate Today (%)", 0, 100, 45) / 100
    inp_buy       = st.slider("Buy Ratio (%)", 0, 100, 50) / 100

with col2:
    st.markdown("**💰 Position & Risk**")
    inp_size      = st.number_input("Avg Position Size (USD)", 100, 200000, 5000, step=100)
    inp_lev       = st.number_input("Leverage Proxy", 0.1, 50.0, 2.0, step=0.1)
    inp_pnl_std   = st.number_input("PnL Std Dev Today (USD)", 0.0, 50000.0, 500.0, step=50.0)
    inp_fee       = st.number_input("Total Fees Paid Today (USD)", 0.0, 5000.0, 10.0, step=1.0)

with col3:
    st.markdown("**📅 Lagged Features (Past Days)**")
    inp_pnl1      = st.number_input("PnL Lag 1 — Yesterday (USD)", -50000, 50000, 100, step=50)
    inp_pnl2      = st.number_input("PnL Lag 2 — 2 Days Ago (USD)", -50000, 50000, 50, step=50)
    inp_tc_lag1   = st.number_input("Trade Count Lag 1 — Yesterday", 0, 500, 10)

with col4:
    st.markdown("**📈 3-Day Rolling Averages**")
    inp_pnl_roll3 = st.number_input("Daily PnL Roll 3 (USD)", -50000, 50000, 100, step=50)
    inp_tc_roll3  = st.number_input("Trade Count Roll 3", 0, 500, 10)
    inp_wr_roll3  = st.slider("Win Rate Roll 3 (%)", 0, 100, 45) / 100
    inp_size_roll3= st.number_input("Avg Size Roll 3 (USD)", 100, 200000, 5000, step=100)

st.divider()

# ── Sentiment dummy columns (matching: sent_Extreme Fear, sent_Fear, etc.) ────
sent_dummies = {f"sent_{s}": 1 if s == inp_sent else 0 for s in SENTIMENT_ORDER}

# ── Assemble feature vector in EXACT order from notebook ──────────────────────
# feature_cols = [
#   'sent_score','trade_count','win_rate_day','avg_size','pnl_std','buy_ratio',
#   'avg_lev','total_fee','daily_pnl_roll3','trade_count_roll3',
#   'win_rate_day_roll3','avg_size_roll3','pnl_lag1','pnl_lag2','trades_lag1',
#   'sent_Extreme Fear','sent_Fear','sent_Neutral','sent_Greed','sent_Extreme Greed'
# ]

feature_vector = [
    SENT_MAP[inp_sent],   # sent_score
    inp_tc,               # trade_count
    inp_wr,               # win_rate_day
    inp_size,             # avg_size
    inp_pnl_std,          # pnl_std
    inp_buy,              # buy_ratio
    inp_lev,              # avg_lev
    inp_fee,              # total_fee
    inp_pnl_roll3,        # daily_pnl_roll3
    inp_tc_roll3,         # trade_count_roll3
    inp_wr_roll3,         # win_rate_day_roll3
    inp_size_roll3,       # avg_size_roll3
    inp_pnl1,             # pnl_lag1
    inp_pnl2,             # pnl_lag2
    inp_tc_lag1,          # trades_lag1
    sent_dummies["sent_Extreme Fear"],
    sent_dummies["sent_Fear"],
    sent_dummies["sent_Neutral"],
    sent_dummies["sent_Greed"],
    sent_dummies["sent_Extreme Greed"],
]

# ── Predict Button ─────────────────────────────────────────────────────────────
if st.button("🔮 Predict Tomorrow's PnL Bucket", use_container_width=True, type="primary"):
    try:
        X_input = np.array([feature_vector])
        prediction = model.predict(X_input)
        pred_label = str(prediction[0])

        emoji_map = {"Profitable": "🟢", "Breakeven": "🟡", "Loss": "🔴"}

        col_res1, col_res2 = st.columns([1, 2])

        with col_res1:
            st.success(f"### {emoji_map.get(pred_label, '🔵')} **{pred_label}**")
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X_input)[0]
                st.metric("Confidence", f"{proba.max()*100:.1f}%")

        with col_res2:
            if hasattr(model, "predict_proba"):
                proba   = model.predict_proba(X_input)[0]
                classes = model.classes_
                colors  = ["#2ca02c" if c == "Profitable"
                           else "#ff7f0e" if c == "Breakeven"
                           else "#d62728" for c in classes]
                fig, ax = plt.subplots(figsize=(7, 3))
                bars = ax.barh(classes, proba * 100, color=colors, edgecolor="black")
                ax.set_xlabel("Probability (%)")
                ax.set_xlim(0, 100)
                ax.set_title("Prediction Confidence")
                for bar, val in zip(bars, proba * 100):
                    ax.text(val + 0.5, bar.get_y() + bar.get_height() / 2,
                            f"{val:.1f}%", va="center", fontsize=10)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
        st.warning("Make sure model.pkl was saved from your notebook after training.")

# ── Feature Reference ──────────────────────────────────────────────────────────
st.divider()
with st.expander("📋 Feature Reference — All 20 Features Used"):
    st.markdown("""
    | # | Feature | Description |
    |---|---|---|
    | 1 | **sent_score** | Sentiment score: Extreme Fear=1, Fear=2, Neutral=3, Greed=4, Extreme Greed=5 |
    | 2 | **trade_count** | Number of trades today |
    | 3 | **win_rate_day** | Fraction of winning trades today (0–1) |
    | 4 | **avg_size** | Average position size in USD today |
    | 5 | **pnl_std** | Standard deviation of PnL today (risk proxy) |
    | 6 | **buy_ratio** | Fraction of BUY trades today (0–1) |
    | 7 | **avg_lev** | Leverage proxy (Size USD / Start Position) |
    | 8 | **total_fee** | Total fees paid today in USD |
    | 9 | **daily_pnl_roll3** | 3-day rolling average of daily PnL |
    | 10 | **trade_count_roll3** | 3-day rolling average of trade count |
    | 11 | **win_rate_day_roll3** | 3-day rolling average of win rate |
    | 12 | **avg_size_roll3** | 3-day rolling average of avg position size |
    | 13 | **pnl_lag1** | Yesterday's total PnL |
    | 14 | **pnl_lag2** | Total PnL from 2 days ago |
    | 15 | **trades_lag1** | Yesterday's trade count |
    | 16–20 | **sent_\*** | One-hot dummies for each sentiment class |
    """)

with st.expander("🤖 Model Info"):
    st.write(f"**Model Type:** `{type(model).__name__}`")
    if hasattr(model, "n_features_in_"):
        st.write(f"**Expected Features:** `{model.n_features_in_}`")
    if hasattr(model, "classes_"):
        st.write(f"**Output Classes:** `{list(model.classes_)}`")
