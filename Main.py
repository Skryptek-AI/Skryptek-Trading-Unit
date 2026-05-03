"""
main.py — AI Trading Agent Entry Point
=======================================
Run modes:
  python main.py --mode train          # train on synthetic data
  python main.py --mode train --csv prices.csv
  python main.py --mode eval           # evaluate saved model
  python main.py --mode live           # paper-trade tick by tick

Data format expected for --csv:
  Columns: timestamp, open, high, low, close, volume
"""

import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os, sys

# ── make imports work regardless of working dir ───────────────────────
sys.path.insert(0, os.path.dirname(__file__))

from environment import TradingEnv
from agent import DQNAgent
from trainer import Trainer, sharpe, max_drawdown


# ─────────────────────────────────────────────
#  SYNTHETIC DATA GENERATOR
# ─────────────────────────────────────────────

def make_synthetic_data(n: int = 2000, seed: int = 42) -> pd.DataFrame:
    """
    GBM + regime shifts + fat-tailed noise mimicking real market behaviour.
    """
    rng   = np.random.default_rng(seed)
    price = 100.0
    rows  = []
    ts    = datetime(2022, 1, 1)
    regime_len = 0
    trend = 0.03      # starting drift

    for i in range(n):
        # regime shift every ~120 bars
        regime_len += 1
        if regime_len > rng.integers(80, 160):
            trend = rng.choice([-0.05, 0.0, 0.05, 0.08])
            regime_len = 0

        vol   = rng.uniform(0.8, 2.5)
        noise = rng.standard_t(df=4) * vol     # fat tails
        ret   = trend / 252 + noise / 100
        open_ = price
        close = max(open_ * (1 + ret), 0.01)
        high  = max(open_, close) * (1 + abs(rng.normal(0, 0.003)))
        low   = min(open_, close) * (1 - abs(rng.normal(0, 0.003)))
        vol_v = rng.uniform(1_000, 20_000) * (1 + abs(ret) * 10)

        rows.append({
            "timestamp": ts,
            "open":  round(open_, 4),
            "high":  round(high,  4),
            "low":   round(low,   4),
            "close": round(close, 4),
            "volume":round(vol_v, 2),
        })
        price = close
        ts   += timedelta(hours=1)

    return pd.DataFrame(rows)


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    required = {"open", "high", "low", "close", "volume"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")
    df = df.sort_values("timestamp").reset_index(drop=True)
    print(f"[Data] Loaded {len(df)} rows from {path}")
    return df


# ─────────────────────────────────────────────
#  TRAIN MODE
# ─────────────────────────────────────────────

def run_train(args):
    if args.csv:
        df = load_csv(args.csv)
    else:
        print("[Data] Generating synthetic market data (2000 bars)…")
        df = make_synthetic_data(2000)

    split = int(len(df) * 0.8)
    train_df = df.iloc[:split].reset_index(drop=True)
    val_df   = df.iloc[split:].reset_index(drop=True)
    print(f"[Data] Train={len(train_df)} bars | Val={len(val_df)} bars")

    trainer = Trainer(
        train_df       = train_df,
        val_df         = val_df,
        window_size    = 20,
        initial_cash   = 10_000.0,
        n_episodes     = args.episodes,
        hidden_size    = 256,
        lr             = 3e-4,
        gamma          = 0.99,
        epsilon_start  = 1.0,
        epsilon_end    = 0.05,
        epsilon_decay  = 0.995,
        batch_size     = 64,
        checkpoint_dir = args.checkpoint_dir,
        log_every      = max(1, args.episodes // 20),
    )

    trainer.train()
    report_path = os.path.join(args.checkpoint_dir, "report.html")
    trainer.generate_report(report_path)
    print(f"\n[Done] Open {report_path} in a browser to view the full report.")


# ─────────────────────────────────────────────
#  EVAL MODE
# ─────────────────────────────────────────────

def run_eval(args):
    if args.csv:
        df = load_csv(args.csv)
    else:
        print("[Data] Using synthetic data for evaluation…")
        df = make_synthetic_data(500, seed=99)

    env   = TradingEnv(df, window_size=20, initial_cash=10_000)
    agent = DQNAgent(obs_dim=env.obs_shape[0], checkpoint_dir=args.checkpoint_dir)
    agent.load("best")
    agent.epsilon = 0.0   # pure exploitation

    obs  = env.reset()
    done = False
    while not done:
        action = agent.act(obs, training=False)
        obs, _, done, info = env.step(action)

    eq = env.equity_curve
    print("\n" + "="*50)
    print("  EVALUATION RESULTS")
    print("="*50)
    print(f"  Final Equity  : ${eq[-1]:,.2f}")
    print(f"  Total PnL     : {info['total_pnl']:+.4f}")
    print(f"  Trades        : {info['total_trades']}")
    print(f"  Win Rate      : {info['win_rate']:.1%}")
    print(f"  Sharpe Ratio  : {sharpe(eq):+.4f}")
    print(f"  Max Drawdown  : {max_drawdown(eq):.1%}")
    print("="*50)

    print("\nLast 10 trades:")
    for t in env.trade_log[-10:]:
        sign = "✓" if t["pnl"] > 0 else "✗"
        print(f"  {sign} {t['reason']:5s}  entry={t['entry']:.4f}"
              f"  exit={t['exit']:.4f}  pnl={t['pnl']:+.4f}")


# ─────────────────────────────────────────────
#  LIVE / PAPER-TRADE MODE
# ─────────────────────────────────────────────

def run_live(args):
    """
    Paper-trade mode: feed one candle at a time via stdin.
    Format: open,high,low,close,volume  (comma-separated)
    Type 'quit' to stop.
    """
    print("[Live] Paper-trade mode. Enter candles as: open,high,low,close,volume")
    print("[Live] Loading agent from checkpoint…")

    # Bootstrap with a small history window so features are valid
    history = list(make_synthetic_data(50, seed=0).itertuples(index=False))
    bootstrap_df = pd.DataFrame(history)

    env   = TradingEnv(bootstrap_df, window_size=20)
    agent = DQNAgent(obs_dim=env.obs_shape[0], checkpoint_dir=args.checkpoint_dir)
    try:
        agent.load("best")
        agent.epsilon = 0.05   # tiny exploration
    except Exception:
        print("[Live] No checkpoint found — agent will act randomly until trained.")

    obs = env.reset()
    step_count = 0

    while True:
        line = input(f"\nCandle {step_count+1} › ").strip()
        if line.lower() in ("quit", "exit", "q"):
            break
        try:
            parts = [float(x) for x in line.split(",")]
            if len(parts) < 4:
                print("Need at least: open,high,low,close")
                continue
            o, h, l, c = parts[:4]
            v = parts[4] if len(parts) > 4 else 1000.0
        except ValueError:
            print("Invalid input. Use: open,high,low,close[,volume]")
            continue

        action = agent.act(obs, training=False)
        labels = ["HOLD", "BUY ", "SELL"]
        obs, reward, done, info = env.step(action)
        step_count += 1

        print(f"  Action : {labels[action]}")
        print(f"  Equity : ${info['equity']:,.2f}  PnL={info['total_pnl']:+.4f}")
        print(f"  Position held: {'YES' if info['position'] > 0 else 'NO'}")

        if done:
            print("[Live] Episode done.")
            break


# ─────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="AI Reinforcement Learning Trading Agent")
    parser.add_argument("--mode",           default="train",
                        choices=["train", "eval", "live"],
                        help="Run mode (default: train)")
    parser.add_argument("--csv",            default=None,
                        help="Path to OHLCV CSV file (optional)")
    parser.add_argument("--episodes",       type=int, default=100,
                        help="Training episodes (default: 100)")
    parser.add_argument("--checkpoint-dir", default="checkpoints",
                        dest="checkpoint_dir",
                        help="Directory to save/load model (default: checkpoints/)")
    args = parser.parse_args()

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    if args.mode == "train":
        run_train(args)
    elif args.mode == "eval":
        run_eval(args)
    elif args.mode == "live":
        run_live(args)


if __name__ == "__main__":
    main()
