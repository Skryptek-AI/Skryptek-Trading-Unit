"""
Trading Environment
===================
OpenAI Gym-compatible environment that simulates a market.
The agent observes a window of OHLCV features + portfolio state,
and chooses: HOLD (0), BUY (1), SELL (2).
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple


class TradingEnv:
    """
    State  : (window_size, n_features) + [position, unrealised_pnl, cash_ratio]
    Actions: 0=HOLD, 1=BUY, 2=SELL
    Reward : shaped reward = realised PnL + hold bonus - trade penalty
    """

    HOLD = 0
    BUY  = 1
    SELL = 2

    def __init__(
        self,
        df:              pd.DataFrame,
        window_size:     int   = 20,
        initial_cash:    float = 10_000.0,
        trade_size:      float = 0.2,       # fraction of cash per trade
        trade_fee:       float = 0.001,     # 0.1% commission
        stop_loss:       float = 0.05,      # 5% stop-loss
        take_profit:     float = 0.10,      # 10% take-profit
        reward_scaling:  float = 100.0,
    ):
        self.df             = df.reset_index(drop=True)
        self.window_size    = window_size
        self.initial_cash   = initial_cash
        self.trade_size     = trade_size
        self.trade_fee      = trade_fee
        self.stop_loss      = stop_loss
        self.take_profit    = take_profit
        self.reward_scaling = reward_scaling

        self._build_features()
        self.n_features     = self.features.shape[1]
        self.obs_shape      = (window_size * self.n_features + 3,)

        self.reset()

    # ── feature engineering ────────────────────────────────────────────

    def _build_features(self):
        df = self.df.copy()
        c  = df["close"]

        df["ret1"]   = c.pct_change(1)
        df["ret5"]   = c.pct_change(5)
        df["ret10"]  = c.pct_change(10)
        df["ema9"]   = c.ewm(span=9,  adjust=False).mean()
        df["ema21"]  = c.ewm(span=21, adjust=False).mean()
        df["ema_cross"] = (df["ema9"] - df["ema21"]) / c

        high, low    = df["high"], df["low"]
        df["hl_ratio"] = (high - low) / c
        df["body"]   = (c - df["open"]) / c

        # RSI
        delta   = c.diff()
        gain    = delta.clip(lower=0).ewm(span=14, adjust=False).mean()
        loss    = (-delta.clip(upper=0)).ewm(span=14, adjust=False).mean()
        rs      = gain / (loss + 1e-9)
        df["rsi"] = (100 - 100 / (1 + rs)) / 100   # normalise to [0,1]

        # ATR (normalised)
        prev_c  = c.shift(1)
        tr      = pd.concat([high - low,
                              (high - prev_c).abs(),
                              (low  - prev_c).abs()], axis=1).max(axis=1)
        df["atr_n"] = tr.ewm(span=14, adjust=False).mean() / c

        # Volume change
        df["vol_chg"] = df["volume"].pct_change(1).clip(-3, 3)

        feature_cols = ["ret1","ret5","ret10","ema_cross",
                        "hl_ratio","body","rsi","atr_n","vol_chg"]
        self.feature_cols = feature_cols
        self.features     = df[feature_cols].fillna(0).values
        self.prices       = df["close"].values

    # ── gym interface ──────────────────────────────────────────────────

    def reset(self) -> np.ndarray:
        self.step_idx     = self.window_size
        self.cash         = self.initial_cash
        self.position     = 0.0      # units held
        self.entry_price  = 0.0
        self.total_trades = 0
        self.wins         = 0
        self.total_pnl    = 0.0
        self.equity_curve = [self.initial_cash]
        self.trade_log    = []
        return self._observe()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        price     = self.prices[self.step_idx]
        prev_price= self.prices[self.step_idx - 1]
        reward    = 0.0
        info      = {}

        # ── position management: stop-loss / take-profit ───────────────
        if self.position > 0:
            chg = (price - self.entry_price) / self.entry_price
            if chg <= -self.stop_loss:
                reward += self._close_position(price, "SL")
                action  = self.HOLD
            elif chg >= self.take_profit:
                reward += self._close_position(price, "TP")
                action  = self.HOLD

        # ── action ────────────────────────────────────────────────────
        if action == self.BUY and self.position == 0:
            units = (self.cash * self.trade_size) / price
            cost  = units * price * (1 + self.trade_fee)
            if cost <= self.cash:
                self.cash        -= cost
                self.position     = units
                self.entry_price  = price
                self.total_trades += 1
                reward -= self.trade_fee * self.reward_scaling  # entry cost

        elif action == self.SELL and self.position > 0:
            reward += self._close_position(price, "SELL")

        elif action == self.HOLD and self.position > 0:
            # small reward for riding a winning position
            unrealised = (price - self.entry_price) / self.entry_price
            reward += unrealised * 0.5

        # ── step forward ──────────────────────────────────────────────
        self.step_idx += 1
        equity = self.cash + self.position * self.prices[self.step_idx - 1]
        self.equity_curve.append(equity)

        done = self.step_idx >= len(self.prices) - 1
        if done and self.position > 0:
            reward += self._close_position(self.prices[-1], "EOD")

        info = {
            "equity":      equity,
            "position":    self.position,
            "total_pnl":   self.total_pnl,
            "total_trades":self.total_trades,
            "win_rate":    self.wins / max(1, self.total_trades),
        }
        return self._observe(), reward * self.reward_scaling, done, info

    # ── helpers ────────────────────────────────────────────────────────

    def _close_position(self, price: float, reason: str) -> float:
        proceeds  = self.position * price * (1 - self.trade_fee)
        pnl       = proceeds - self.position * self.entry_price
        self.cash += proceeds
        self.total_pnl += pnl
        if pnl > 0:
            self.wins += 1
        self.trade_log.append({
            "reason":      reason,
            "entry":       self.entry_price,
            "exit":        price,
            "pnl":         round(pnl, 4),
            "units":       round(self.position, 6),
        })
        self.position    = 0.0
        self.entry_price = 0.0
        return pnl / self.initial_cash     # normalised reward

    def _observe(self) -> np.ndarray:
        window  = self.features[self.step_idx - self.window_size : self.step_idx]
        price   = self.prices[self.step_idx - 1]
        equity  = self.cash + self.position * price

        pos_flag   = 1.0 if self.position > 0 else 0.0
        unreal_pnl = ((price - self.entry_price) / self.entry_price
                      if self.position > 0 else 0.0)
        cash_ratio = self.cash / equity if equity > 0 else 1.0

        portfolio_state = np.array([pos_flag, unreal_pnl, cash_ratio],
                                   dtype=np.float32)
        return np.concatenate([window.flatten(), portfolio_state]).astype(np.float32)
