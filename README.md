# ⚡ AI Reinforcement Learning Trading Agent

A self-learning market trading agent built with **Deep Q-Networks (DQN)** and
**candlestick feature engineering**. The agent learns profitable trading strategies
purely by trial and error — no rules are hard-coded.

---

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    TRADING ENVIRONMENT                    │
│  OHLCV data → Feature Engineering → State Observation   │
│  Actions: HOLD / BUY / SELL                               │
│  Rewards: Realised PnL + unrealised bonus - trade fees   │
│  Risk Mgmt: ATR stop-loss + take-profit per position     │
└───────────────────────────┬──────────────────────────────┘
                            │ obs, reward
                            ▼
┌──────────────────────────────────────────────────────────┐
│                      DQN AGENT                            │
│  • Dueling DQN  (V(s) + A(s,a) architecture)             │
│  • Double DQN   (reduces overestimation)                  │
│  • Prioritised Experience Replay                          │
│  • Epsilon-greedy exploration → exploitation decay        │
│  • Periodic target-network sync                           │
│  • Falls back to Q-table if PyTorch is not installed      │
└──────────────────────────────────────────────────────────┘
```

## Features Learned From

| Feature      | Description                        |
|--------------|------------------------------------|
| ret1/5/10    | 1, 5, 10-bar returns               |
| ema_cross    | EMA(9) vs EMA(21) divergence       |
| rsi          | Relative Strength Index (0–1)      |
| atr_n        | Normalised Average True Range      |
| body         | Candle body as % of close          |
| hl_ratio     | High-Low range as % of close       |
| vol_chg      | Volume change %                    |
| position     | Whether agent currently holds      |
| unrealised   | Current open trade P&L             |
| cash_ratio   | Cash / total equity                |

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train (synthetic data, 100 episodes)
python main.py --mode train --episodes 100

# 3. Evaluate the best saved model
python main.py --mode eval

# 4. Train on your own OHLCV CSV
python main.py --mode train --csv my_prices.csv --episodes 200

# 5. Paper-trade interactively (enter candles manually)
python main.py --mode live
```

After training, open `checkpoints/report.html` in a browser for the full
performance dashboard with equity curves, episode PnL, win rate, and trade log.

---

## CSV Format

If supplying your own data:

```
timestamp,open,high,low,close,volume
2024-01-01 00:00:00,100.0,102.5,99.1,101.3,5000
2024-01-01 01:00:00,101.3,103.0,100.8,102.1,6200
...
```

---

## Key Hyperparameters

| Parameter        | Default | Effect                                   |
|------------------|---------|------------------------------------------|
| `--episodes`     | 100     | More = better learning, slower           |
| `window_size`    | 20      | How many candles the agent sees at once  |
| `epsilon_decay`  | 0.995   | Lower = faster shift to exploitation     |
| `gamma`          | 0.99    | Higher = more weight on future rewards   |
| `risk_reward`    | 2.0     | TP = 2× stop-loss distance               |

---

## Connecting to Real Data

**Yahoo Finance (free):**
```python
import yfinance as yf
df = yf.download("BTC-USD", period="1y", interval="1h")
df = df.rename(columns=str.lower)[["open","high","low","close","volume"]]
df["timestamp"] = df.index
df.to_csv("btc_1h.csv", index=False)
```

**Crypto (via CCXT):**
```python
import ccxt, pandas as pd
exchange = ccxt.binance()
ohlcv = exchange.fetch_ohlcv("BTC/USDT", "1h", limit=1000)
df = pd.DataFrame(ohlcv, columns=["timestamp","open","high","low","close","volume"])
df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
```

---

> ⚠️ **Disclaimer**: This is for educational purposes only.
> Past simulated performance does not guarantee real profits.
> Always paper-trade first and never risk money you can't afford to lose.
