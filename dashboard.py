import streamlit as st
import pickle
import gdown
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Trading Sentiment Predictor",
    page_icon="📈",
    layout="wide"
)

# ── Load Model ─────────────────────────────────────────────────────────────────
@st.cache_resource
def load_bundle():
    model_path = "model.pkl"
    # Always re-download fresh copy
    if os.path.exists(model_path):
        os.remove(model_path)

    with st.spinner("Downloading model... ⏳"):
        url = "https://drive.google.com/uc?id=1rrKhnksQSI9su-hwGznF-QZ1ACSiDr9x"
        gdown.download(url, model_path, quiet=False)

    with open(model_path, "rb") as f:
        obj = pickle.load(f)

    return obj  # return raw object first so we can inspect it

raw_obj = load_bundle()

# ── Debug: show what's inside the pkl ─────────────────────────────────────────
with st.expander("🔍 Debug — What's inside model.pkl?", expanded=True):
    st.write(f"**Type:** `{type(raw_obj)}`")
    if isinstance(raw_obj, dict):
        st.write(f"**Keys:** `{list(raw_obj.keys())}`")
        for k, v in raw_obj.items():
            st.write(f"  `{k}` → `{type(v)}`")
    else:
        st.write(f"**Has .predict:** `{hasattr(raw_obj, 'predict')}`")
        st.write(f"**Object:** `{raw_obj}`")

# ── Parse the object ───────────────────────────────────────────────────────────
from sklearn.preprocessing import LabelEncoder

DEFAULT_FEATURES = [
    'sent_score','trade_count','win_rate_day','buy_ratio',
    'avg_size','avg_lev','pnl_std','total_fee',
    'pnl_lag1','pnl_lag2','trades_lag1',
    'daily_pnl_roll3','trade_count_roll3','win_rate_day_roll3','avg_size_roll3',
    'sent_Extreme Fear','sent_Fear','sent_Neutral','sent_Greed','sent_Extreme Greed',
]

def make_default_le():
    le = LabelEncoder()
    le.classes_ = np.array(['Breakeven', 'Loss', 'Profitable'])
    return le

if isinstance(raw_obj, dict):
    # Try every possible key name
    model = (raw_obj.get('model') or raw_obj.get('best_model') or
             raw_obj.get('clf')   or raw_obj.get('estimator') or
             next((v for v in raw_obj.values() if hasattr(v, 'predict')), None))

    le           = raw_obj.get('le', make_default_le())
    feature_cols = raw_obj.get('feature_cols', DEFAULT_FEATURES)
    classes      = raw_obj.get('classes', list(le.classes_))

elif hasattr(raw_obj, 'predict'):
    model        = raw_obj
    le           = make_default_le()
    feature_cols = DEFAULT_FEATURES
    classes      = list(le.classes_)
else:
    st.error("❌ Cannot find a model with .predict() in your pkl file.")
    st.stop()

if model is None:
    st.error("❌ No object with `.predict()` found in the dict. Check debug section above.")
    st.stop()

st.success(f"✅ Model loaded: `{type(model).__name__}`")

SENTIMENT_ORDER = ["Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"]
SENT_MAP        = {"Extreme Fear": 1, "Fear": 2, "Neutral": 3, "Greed": 4, "Extreme Greed": 5}

# ── Header ─────────────────────────────────────────────────────────────────────
st.title("📈 Next-Day Profitability Predictor")
st.markdown("Enter **today's** trading conditions to predict **tomorrow's** PnL outcome.")
st.divider()

# ── Input Features ─────────────────────────────────────────────────────────────
st.subheader("📥 Input Features")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("**📊 Today's Sentiment & Activity**")
    inp_sent  = st.selectbox("Market Sentiment", SENTIMENT_ORDER, index=2)
    inp_tc    = st.number_input("Trade Count", min_value=1, max_value=500, value=10)
    inp_wr    = st.slider("Win Rate Today (%)", 0, 100, 45) / 100
    inp_buy   = st.slider("Buy Ratio (%)", 0, 100, 50) / 100

with col2:
    st.markdown("**💰 Position & Risk**")
    inp_size    = st.number_input("Avg Position Size (USD)", 100, 200000, 5000, step=100)
    inp_lev     = st.number_input("Leverage Proxy", 0.1, 50.0, 2.0, step=0.1)
    inp_pnl_std = st.number_input("PnL Std Dev Today (USD)", 0.0, 50000.0, 500.0, step=50.0)
    inp_fee     = st.number_input("Total Fees Paid Today (USD)", 0.0, 5000.0, 10.0, step=1.0)

with col3:
    st.markdown("**📅 Lagged Features (Past Days)**")
    inp_pnl1    = st.number_input("PnL Lag 1 — Yesterday (USD)", -50000, 50000, 100, step=50)
    inp_pnl2    = st.number_input("PnL Lag 2 — 2 Days Ago (USD)", -50000, 50000, 50, step=50)
    inp_tc_lag1 = st.number_input("Trade Count Lag 1 — Yesterday", 0, 500, 10)

with col4:
    st.markdown("**📈 3-Day Rolling Averages**")
    inp_pnl_roll3  = st.number_input("Daily PnL Roll 3 (USD)", -50000, 50000, 100, step=50)
    inp_tc_roll3   = st.number_input("Trade Count Roll 3", 0, 500, 10)
    inp_wr_roll3   = st.slider("Win Rate Roll 3 (%)", 0, 100, 45) / 100
    inp_size_roll3 = st.number_input("Avg Size Roll 3 (USD)", 100, 200000, 5000, step=100)

st.divider()

# ── Build feature vector ───────────────────────────────────────────────────────
sent_dummy = {f"sent_{s}": (1 if s == inp_sent else 0) for s in SENTIMENT_ORDER}

feature_vector = [
    SENT_MAP[inp_sent], inp_tc, inp_wr, inp_buy,
    inp_size, inp_lev, inp_pnl_std, inp_fee,
    inp_pnl1, inp_pnl2, inp_tc_lag1,
    inp_pnl_roll3, inp_tc_roll3, inp_wr_roll3, inp_size_roll3,
    sent_dummy["sent_Extreme Fear"], sent_dummy["sent_Fear"],
    sent_dummy["sent_Neutral"],      sent_dummy["sent_Greed"],
    sent_dummy["sent_Extreme Greed"],
]

# ── Predict ────────────────────────────────────────────────────────────────────
if st.button("🔮 Predict Tomorrow's PnL Bucket", use_container_width=True, type="primary"):
    try:
        X_input    = np.array([feature_vector])
        pred_enc   = model.predict(X_input)[0]
        try:
            pred_label = le.inverse_transform([pred_enc])[0]
        except Exception:
            pred_label = str(pred_enc)

        proba     = model.predict_proba(X_input)[0]
        emoji_map = {"Profitable": "🟢", "Breakeven": "🟡", "Loss": "🔴"}

        col_r1, col_r2 = st.columns([1, 2])
        with col_r1:
            st.success(f"### {emoji_map.get(pred_label,'🔵')} **{pred_label}**")
            st.metric("Confidence", f"{proba.max()*100:.1f}%")
            for cls, p in zip(classes, proba):
                st.write(f"{emoji_map.get(cls,'')} {cls}: **{p*100:.1f}%**")

        with col_r2:
            color_map  = {"Profitable":"#2ca02c","Breakeven":"#ff7f0e","Loss":"#d62728"}
            bar_colors = [color_map.get(c,"#4c72b0") for c in classes]
            fig, ax    = plt.subplots(figsize=(7, 3))
            bars       = ax.barh(classes, proba*100, color=bar_colors, edgecolor="black")
            ax.set_xlabel("Probability (%)"); ax.set_xlim(0, 100)
            ax.set_title("Prediction Confidence")
            for bar, val in zip(bars, proba*100):
                ax.text(val+0.5, bar.get_y()+bar.get_height()/2,
                        f"{val:.1f}%", va="center", fontsize=11, fontweight="bold")
            plt.tight_layout(); st.pyplot(fig); plt.close()

    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
