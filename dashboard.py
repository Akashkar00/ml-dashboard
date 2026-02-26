import streamlit as st
import pickle
import gdown
import os
import numpy as np
import pandas as pd

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Trading Sentiment Predictor",
    page_icon="📈",
    layout="centered"
)

# ── Load Model from Google Drive ───────────────────────────────────────────────
@st.cache_resource
def load_model():
    model_path = "model.pkl"
    if not os.path.exists(model_path):
        with st.spinner("Downloading model... please wait ⏳"):
            url = "https://drive.google.com/uc?id=1dYBJUze-GVt_2-lDbAE2fCKXXW04CLI_"
            gdown.download(url, model_path, quiet=False)
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

SENTIMENT_ORDER = ["Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"]
sent_map        = {"Extreme Fear": 1, "Fear": 2, "Neutral": 3, "Greed": 4, "Extreme Greed": 5}

# ── Header ─────────────────────────────────────────────────────────────────────
st.title("📈 Next-Day Profitability Predictor")
st.markdown("Enter **today's** trading conditions to predict **tomorrow's** likely PnL bucket.")
st.divider()

# ── Input Features ─────────────────────────────────────────────────────────────
st.subheader("📥 Input Features")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**📊 Sentiment & Activity**")
    inp_sent = st.selectbox("Today's Market Sentiment", SENTIMENT_ORDER, index=2)
    inp_wr   = st.slider("Today Win Rate (%)", 0, 100, 45) / 100
    inp_tc   = st.number_input("Today Trade Count", min_value=1, max_value=500, value=10)

with col2:
    st.markdown("**💰 Position & Leverage**")
    inp_size = st.number_input("Avg Position Size (USD)", min_value=100, max_value=100000,
                                value=5000, step=100)
    inp_lev  = st.number_input("Leverage Proxy", min_value=0.1, max_value=50.0,
                                value=2.0, step=0.1)
    inp_buy  = st.slider("Buy Ratio (%)", 0, 100, 50) / 100

with col3:
    st.markdown("**📅 Historical Context**")
    inp_pnl1 = st.number_input("PnL Lag 1 — Yesterday (USD)",
                                min_value=-10000, max_value=10000, value=100, step=50)
    inp_pnl2 = st.number_input("PnL Lag 2 — 2 Days Ago (USD)",
                                min_value=-10000, max_value=10000, value=50, step=50)
    inp_fee  = st.number_input("Total Fees Paid Today (USD)",
                                min_value=0.0, max_value=1000.0, value=10.0, step=1.0)

st.divider()

# ── Predict Button ─────────────────────────────────────────────────────────────
if st.button("🔮 Predict Tomorrow's PnL Bucket", use_container_width=True, type="primary"):
    try:
        input_data = np.array([[
            sent_map[inp_sent],   # sent_score
            inp_tc,               # trade_count
            inp_wr,               # win_rate_day
            inp_size,             # avg_size
            inp_lev,              # avg_lev
            inp_buy,              # buy_ratio
            inp_fee,              # total_fee
            inp_pnl1,             # pnl_lag1
            inp_pnl2,             # pnl_lag2
        ]])

        prediction = model.predict(input_data)
        emoji_map  = {"Profitable": "🟢", "Breakeven": "🟡", "Loss": "🔴"}
        pred_label = str(prediction[0])

        st.success(f"### {emoji_map.get(pred_label, '🔵')} Predicted: **{pred_label}**")

        # Probabilities bar chart
        if hasattr(model, "predict_proba"):
            proba   = model.predict_proba(input_data)[0]
            classes = model.classes_

            st.subheader("📊 Prediction Confidence")
            proba_df = pd.DataFrame({
                "Outcome":     classes,
                "Probability": (proba * 100).round(1)
            }).set_index("Outcome")

            colors = []
            for c in classes:
                if c == "Profitable":   colors.append("#2ca02c")
                elif c == "Breakeven":  colors.append("#ff7f0e")
                else:                   colors.append("#d62728")

            import matplotlib.pyplot as plt
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

            st.metric("Top Confidence", f"{proba.max()*100:.1f}%")

    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
        st.warning("Make sure the model was trained with these exact 9 features in this order.")

# ── Model Info ─────────────────────────────────────────────────────────────────
st.divider()
with st.expander("ℹ️ Feature Reference"):
    st.markdown("""
    | # | Feature | Description |
    |---|---|---|
    | 1 | **sent_score** | Sentiment mapped: Extreme Fear=1 → Extreme Greed=5 |
    | 2 | **trade_count** | Number of trades made today |
    | 3 | **win_rate_day** | Fraction of winning trades today (0–1) |
    | 4 | **avg_size** | Average position size in USD |
    | 5 | **avg_lev** | Leverage proxy (Size / Start Position) |
    | 6 | **buy_ratio** | Fraction of BUY trades today (0–1) |
    | 7 | **total_fee** | Total fees paid today in USD |
    | 8 | **pnl_lag1** | Yesterday's total PnL |
    | 9 | **pnl_lag2** | Total PnL from 2 days ago |
    """)

with st.expander("🤖 Model Info"):
    st.write(f"**Model Type:** `{type(model).__name__}`")
    if hasattr(model, "n_features_in_"):
        st.write(f"**Expected Features:** `{model.n_features_in_}`")
    if hasattr(model, "classes_"):
        st.write(f"**Output Classes:** `{list(model.classes_)}`")
