# 📈 Trading × Sentiment Analysis Dashboard

A machine learning dashboard that predicts **next-day trading profitability** based on market sentiment (Fear & Greed Index) and trader behaviour features.

Built with **Streamlit**, **Scikit-learn**, **Matplotlib**, and **Seaborn**.

---

## 🚀 Live Demo

👉 [Click here to open the dashboard](https://akash-kar-ie5hf2qhrvnrxccz3pjmbt.streamlit.app/)
> *(Replace with your actual Streamlit Cloud URL after deploying)*

---

## 📁 Project Structure

```
📦 your-repo/
├── dashboard.py              ← Main Streamlit app
├── trading_analysis.ipynb      ← Jupyter notebook (model building & training)
├── requirements.txt          ← Python dependencies
├── README.md                 ← Project documentation
└── model.pkl                 ← ⚠️ NOT uploaded (too large, loads from Google Drive)
```

---

## ⚙️ Setup & How to Run

### Option 1 — Run Locally

**Step 1 — Clone the repo**
```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME
```

**Step 2 — Install dependencies**
```bash
pip install -r requirements.txt
```

**Step 3 — Run the app**
```bash
streamlit run dashboard.py
```

**Step 4 — Open in browser**
```
http://localhost:8501
```

> ✅ The model downloads automatically from Google Drive on first run.

---

### Option 2 — Deploy on Streamlit Cloud (Free)

1. Push this repo to **GitHub**
2. Go to **[share.streamlit.io](https://share.streamlit.io)**
3. Click **"New app"**
4. Select your GitHub repo
5. Set **Main file path** → `dashboard.py`
6. Click **Deploy** ✅

---

## 🤖 How the Model Works

### Input Features (20 total)

| Group | Features |
|---|---|
| 📊 Sentiment & Activity | Market Sentiment, Trade Count, Win Rate, Buy Ratio |
| 💰 Position & Risk | Avg Position Size, Leverage Proxy, PnL Std Dev, Total Fees |
| 📅 Lagged (Past Days) | PnL Lag 1, PnL Lag 2, Trade Count Lag 1 |
| 📈 3-Day Rolling Avgs | Rolling PnL, Rolling Trade Count, Rolling Win Rate, Rolling Avg Size |
| 🔢 Sentiment Dummies | One-hot encoding for each sentiment class |

### Output (Prediction)
The model predicts **tomorrow's PnL bucket**:
- 🟢 **Profitable** — next day PnL > $50
- 🟡 **Breakeven** — next day PnL between -$50 and $50
- 🔴 **Loss** — next day PnL < -$50

### Models Trained & Compared
| Model | Accuracy | AUC-ROC |
|---|---|---|
| Logistic Regression | 0.331 | 0.591 |
| Random Forest | 0.578 | 0.645 |
| Gradient Boosting | 0.541 | 0.623 |

> Best model selected automatically based on AUC-ROC after hyperparameter tuning.

---

## 📊 Data Sources

| Dataset | Description |
|---|---|
| `fear_greed_index.csv` | Daily Fear & Greed Index (value + classification) |
| `historical_data.csv` | Hyperliquid on-chain trading data (211,000+ trades) |

---

## 📦 Requirements

```
streamlit
scikit-learn
pandas
numpy
matplotlib
seaborn
gdown
scipy
```

Install all at once:
```bash
pip install -r requirements.txt
```

---

## 🗂️ How to Regenerate model.pkl

1. Open `trading_analysis.ipynb` in Jupyter or Google Colab
2. Upload `fear_greed_index.csv` and `historical_data.csv`
3. Run all cells top to bottom
4. The last cell saves `model.pkl`
5. Upload `model.pkl` to Google Drive
6. Set sharing to **"Anyone with the link"**
7. Replace the File ID in `dashboard.py`:
```python
url = "https://drive.google.com/uc?id=YOUR_NEW_FILE_ID"
```

---

## 💡 Key Insights

1. **Fear regime = higher PnL variance** — winners win bigger, losers lose bigger
2. **Extreme Greed = over-trading** — most trades/day but smallest position sizes
3. **Consistent Winners are regime-agnostic** — they perform well across all sentiments

## 📋 Strategy Rules

| Segment | Fear Regime | Greed Regime |
|---|---|---|
| High Leverage | ↓ size 35%, max 3 trades/day | Normal |
| Frequent Trader | ↓ frequency 20% | ↓ frequency 40%, prefer SHORT |
| Consistent Winner | Maintain strategy | Maintain strategy |

---

## 👤 Author

**Akash**
B.Tech — Biotechnology & Medical Engineering, NIT Rourkela

---

## 📄 License

MIT License — free to use and modify.
