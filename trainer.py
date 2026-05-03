"""
Trainer
=======
Runs the training loop:
  - Episode = one pass through price history
  - Agent explores early (high epsilon), exploits later
  - Tracks equity curve, win rate, Sharpe ratio
  - Saves best model by total PnL
  - Produces a final HTML performance report
"""

import numpy as np
import pandas as pd
import os
import json
from datetime import datetime
from typing import Optional

from environment import TradingEnv
from agent import DQNAgent


# ─────────────────────────────────────────────
#  METRICS
# ─────────────────────────────────────────────

def sharpe(equity_curve: list, risk_free: float = 0.0) -> float:
    arr  = np.array(equity_curve)
    rets = np.diff(arr) / arr[:-1]
    if rets.std() == 0:
        return 0.0
    return float((rets.mean() - risk_free) / rets.std() * np.sqrt(252))


def max_drawdown(equity_curve: list) -> float:
    arr     = np.array(equity_curve)
    peak    = np.maximum.accumulate(arr)
    dd      = (arr - peak) / peak
    return float(dd.min())


# ─────────────────────────────────────────────
#  TRAINER
# ─────────────────────────────────────────────

class Trainer:

    def __init__(
        self,
        train_df:       pd.DataFrame,
        val_df:         Optional[pd.DataFrame] = None,
        window_size:    int   = 20,
        initial_cash:   float = 10_000.0,
        n_episodes:     int   = 100,
        hidden_size:    int   = 256,
        lr:             float = 3e-4,
        gamma:          float = 0.99,
        epsilon_start:  float = 1.0,
        epsilon_end:    float = 0.05,
        epsilon_decay:  float = 0.995,
        batch_size:     int   = 64,
        checkpoint_dir: str   = "checkpoints",
        log_every:      int   = 10,
    ):
        self.n_episodes     = n_episodes
        self.log_every      = log_every
        self.checkpoint_dir = checkpoint_dir

        self.train_env = TradingEnv(
            train_df, window_size=window_size, initial_cash=initial_cash
        )
        self.val_env = (TradingEnv(val_df, window_size=window_size,
                                   initial_cash=initial_cash)
                        if val_df is not None else None)

        obs_dim = self.train_env.obs_shape[0]
        self.agent = DQNAgent(
            obs_dim        = obs_dim,
            n_actions      = 3,
            lr             = lr,
            gamma          = gamma,
            epsilon_start  = epsilon_start,
            epsilon_end    = epsilon_end,
            epsilon_decay  = epsilon_decay,
            batch_size     = batch_size,
            hidden_size    = hidden_size,
            checkpoint_dir = checkpoint_dir,
        )

        self.history = []        # per-episode stats
        self.best_pnl = -np.inf

    # ── main training loop ─────────────────────────────────────────────

    def train(self):
        print(f"\n{'='*60}")
        print(f"  AI Trading Agent — Training for {self.n_episodes} episodes")
        print(f"{'='*60}\n")

        for ep in range(1, self.n_episodes + 1):
            obs  = self.train_env.reset()
            done = False
            ep_reward = 0.0
            losses    = []

            while not done:
                action              = self.agent.act(obs, training=True)
                next_obs, reward, done, info = self.train_env.step(action)
                self.agent.remember(obs, action, reward, next_obs, done)
                loss = self.agent.learn()
                if loss:
                    losses.append(loss)
                obs        = next_obs
                ep_reward += reward

            self.agent.decay_epsilon()

            # ── collect episode stats ──────────────────────────────────
            eq     = self.train_env.equity_curve
            stats  = {
                "episode":    ep,
                "reward":     round(ep_reward, 4),
                "total_pnl":  round(info["total_pnl"], 4),
                "trades":     info["total_trades"],
                "win_rate":   round(info["win_rate"], 4),
                "sharpe":     round(sharpe(eq), 4),
                "max_dd":     round(max_drawdown(eq), 4),
                "final_equity": round(eq[-1], 2),
                "epsilon":    round(self.agent.epsilon, 4),
                "avg_loss":   round(float(np.mean(losses)), 6) if losses else 0,
            }
            self.history.append(stats)

            # ── save best model ────────────────────────────────────────
            if info["total_pnl"] > self.best_pnl:
                self.best_pnl = info["total_pnl"]
                self.agent.save("best")

            # ── log progress ───────────────────────────────────────────
            if ep % self.log_every == 0 or ep == 1:
                val_info = self._validate() if self.val_env else {}
                val_str  = (f"  val_pnl={val_info.get('pnl',0):+.2f}"
                            f"  val_wr={val_info.get('win_rate',0):.1%}"
                            if val_info else "")
                print(
                    f"Ep {ep:4d}/{self.n_episodes}"
                    f"  pnl={stats['total_pnl']:+8.2f}"
                    f"  trades={stats['trades']:3d}"
                    f"  win={stats['win_rate']:.1%}"
                    f"  sharpe={stats['sharpe']:+.2f}"
                    f"  dd={stats['max_dd']:.1%}"
                    f"  ε={stats['epsilon']:.3f}"
                    f"  loss={stats['avg_loss']:.5f}"
                    + val_str
                )

        self.agent.save("final")
        self._save_history()
        print(f"\n[Trainer] Training complete. Best PnL = {self.best_pnl:+.4f}")
        return self.history

    # ── validation pass (no exploration) ──────────────────────────────

    def _validate(self) -> dict:
        obs  = self.val_env.reset()
        done = False
        while not done:
            action = self.agent.act(obs, training=False)
            obs, _, done, info = self.val_env.step(action)
        return {
            "pnl":      info["total_pnl"],
            "trades":   info["total_trades"],
            "win_rate": info["win_rate"],
            "sharpe":   sharpe(self.val_env.equity_curve),
        }

    # ── persistence ────────────────────────────────────────────────────

    def _save_history(self):
        path = os.path.join(self.checkpoint_dir, "training_history.json")
        with open(path, "w") as f:
            json.dump(self.history, f, indent=2)
        print(f"[Trainer] History → {path}")

    # ── HTML report ────────────────────────────────────────────────────

    def generate_report(self, output_path: str = "report.html"):
        df = pd.DataFrame(self.history)
        val_result = self._validate() if self.val_env else {}

        best = df.loc[df["total_pnl"].idxmax()]
        last = df.iloc[-1]

        eq_train = self.train_env.equity_curve
        eq_json  = json.dumps([round(v, 2) for v in eq_train[::max(1, len(eq_train)//300)]])
        pnl_json = json.dumps(list(df["total_pnl"]))
        wr_json  = json.dumps(list(df["win_rate"]))
        eps_json = json.dumps(list(df["epsilon"]))
        ep_json  = json.dumps(list(df["episode"]))

        trade_rows = ""
        for t in self.train_env.trade_log[-20:]:
            color = "#00e676" if t["pnl"] > 0 else "#ff5252"
            trade_rows += (
                f"<tr>"
                f"<td>{t['reason']}</td>"
                f"<td>${t['entry']:.4f}</td>"
                f"<td>${t['exit']:.4f}</td>"
                f"<td style='color:{color}'>{t['pnl']:+.4f}</td>"
                f"</tr>"
            )

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>AI Trading Agent — Report</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@400;600;700&display=swap');
  :root {{
    --bg: #080c10; --surface: #0d1117; --border: #1c2730;
    --accent: #00e5ff; --green: #00e676; --red: #ff5252;
    --text: #c9d1d9; --muted: #586069;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: var(--bg); color: var(--text); font-family: 'Rajdhani', sans-serif;
         font-size: 15px; line-height: 1.6; padding: 32px; }}
  h1 {{ font-size: 2rem; color: var(--accent); letter-spacing: 3px;
       text-transform: uppercase; border-bottom: 1px solid var(--border);
       padding-bottom: 12px; margin-bottom: 24px; font-weight: 700; }}
  h2 {{ font-size: 1.1rem; color: var(--accent); letter-spacing: 2px;
       text-transform: uppercase; margin-bottom: 16px; font-weight: 600; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
           gap: 16px; margin-bottom: 32px; }}
  .card {{ background: var(--surface); border: 1px solid var(--border);
           border-radius: 6px; padding: 20px; }}
  .card .label {{ font-size: 0.75rem; color: var(--muted); letter-spacing: 2px;
                  text-transform: uppercase; margin-bottom: 6px; }}
  .card .value {{ font-size: 1.6rem; font-family: 'Share Tech Mono', monospace;
                  font-weight: 700; }}
  .green {{ color: var(--green); }}
  .red   {{ color: var(--red); }}
  .cyan  {{ color: var(--accent); }}
  .chart-wrap {{ background: var(--surface); border: 1px solid var(--border);
                 border-radius: 6px; padding: 24px; margin-bottom: 24px; }}
  .charts-row {{ display: grid; grid-template-columns: 1fr 1fr; gap: 24px; margin-bottom: 24px; }}
  table {{ width: 100%; border-collapse: collapse; font-family: 'Share Tech Mono', monospace;
           font-size: 0.85rem; }}
  th {{ color: var(--muted); text-align: left; padding: 8px 12px;
        border-bottom: 1px solid var(--border); letter-spacing: 1px;
        text-transform: uppercase; font-family: 'Rajdhani', sans-serif; }}
  td {{ padding: 8px 12px; border-bottom: 1px solid #111; }}
  tr:hover td {{ background: #0d1117; }}
  .timestamp {{ color: var(--muted); font-size: 0.78rem; margin-bottom: 32px;
                font-family: 'Share Tech Mono', monospace; }}
</style>
</head>
<body>
<h1>⚡ AI Trading Agent — Performance Report</h1>
<p class="timestamp">Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
   &nbsp;|&nbsp; {self.n_episodes} training episodes</p>

<div class="grid">
  <div class="card">
    <div class="label">Best Episode PnL</div>
    <div class="value {'green' if best['total_pnl']>0 else 'red'}">{best['total_pnl']:+.2f}</div>
  </div>
  <div class="card">
    <div class="label">Final Episode PnL</div>
    <div class="value {'green' if last['total_pnl']>0 else 'red'}">{last['total_pnl']:+.2f}</div>
  </div>
  <div class="card">
    <div class="label">Best Sharpe</div>
    <div class="value cyan">{df['sharpe'].max():+.3f}</div>
  </div>
  <div class="card">
    <div class="label">Max Win Rate</div>
    <div class="value green">{df['win_rate'].max():.1%}</div>
  </div>
  <div class="card">
    <div class="label">Best Max DD</div>
    <div class="value red">{df['max_dd'].min():.1%}</div>
  </div>
  <div class="card">
    <div class="label">Final Equity</div>
    <div class="value cyan">${last['final_equity']:,.0f}</div>
  </div>
</div>

<div class="chart-wrap">
  <h2>Equity Curve — Best Training Episode</h2>
  <canvas id="eqChart" height="80"></canvas>
</div>

<div class="charts-row">
  <div class="chart-wrap">
    <h2>Episode PnL</h2>
    <canvas id="pnlChart" height="140"></canvas>
  </div>
  <div class="chart-wrap">
    <h2>Win Rate &amp; Epsilon</h2>
    <canvas id="wrChart" height="140"></canvas>
  </div>
</div>

<div class="chart-wrap">
  <h2>Last 20 Trades</h2>
  <table>
    <tr><th>Reason</th><th>Entry</th><th>Exit</th><th>PnL</th></tr>
    {trade_rows}
  </table>
</div>

<script>
const ep   = {ep_json};
const pnl  = {pnl_json};
const wr   = {wr_json};
const eps  = {eps_json};
const eq   = {eq_json};

const cfg = (label, data, color, labels) => ({{
  type:'line', data:{{labels, datasets:[{{label, data, borderColor:color,
  borderWidth:1.5, pointRadius:0, fill:false, tension:0.3}}]}},
  options:{{plugins:{{legend:{{labels:{{color:'#c9d1d9', font:{{family:'Rajdhani'}}}}}}}},
  scales:{{x:{{ticks:{{color:'#586069', maxTicksLimit:10}}, grid:{{color:'#1c2730'}}}},
           y:{{ticks:{{color:'#586069'}}, grid:{{color:'#1c2730'}}}}}}}}
}});

const eqLabels = Array.from({{length:eq.length}},(_,i)=>i);
new Chart('eqChart', cfg('Equity','placeholder','#00e5ff','placeholder'));
const eqCtx = document.getElementById('eqChart').getContext('2d');
const grad  = eqCtx.createLinearGradient(0,0,0,300);
grad.addColorStop(0,'rgba(0,229,255,0.3)');
grad.addColorStop(1,'rgba(0,229,255,0)');
new Chart('eqChart',{{type:'line',data:{{labels:eqLabels,datasets:[{{
  label:'Equity',data:eq,borderColor:'#00e5ff',borderWidth:1.5,
  pointRadius:0,fill:true,backgroundColor:grad,tension:0.2}}]}},
  options:{{plugins:{{legend:{{labels:{{color:'#c9d1d9'}}}}}},
  scales:{{x:{{ticks:{{color:'#586069', maxTicksLimit:10}},grid:{{color:'#1c2730'}}}},
           y:{{ticks:{{color:'#586069'}},grid:{{color:'#1c2730'}}}}}}}}
}});

new Chart('pnlChart',{{type:'bar',data:{{labels:ep,datasets:[{{
  label:'PnL',data:pnl,
  backgroundColor:pnl.map(v=>v>=0?'rgba(0,230,118,0.6)':'rgba(255,82,82,0.6)'),
  borderWidth:0}}]}},
  options:{{plugins:{{legend:{{labels:{{color:'#c9d1d9'}}}}}},
  scales:{{x:{{ticks:{{color:'#586069',maxTicksLimit:12}},grid:{{color:'#1c2730'}}}},
           y:{{ticks:{{color:'#586069'}},grid:{{color:'#1c2730'}}}}}}}}
}});

new Chart('wrChart',{{type:'line',data:{{labels:ep,datasets:[
  {{label:'Win Rate',data:wr.map(v=>+(v*100).toFixed(1)),borderColor:'#00e676',
    borderWidth:1.5,pointRadius:0,tension:0.3,yAxisID:'y'}},
  {{label:'Epsilon',data:eps.map(v=>+(v*100).toFixed(1)),borderColor:'#ff9800',
    borderWidth:1,pointRadius:0,tension:0.1,yAxisID:'y'}}
]}},options:{{plugins:{{legend:{{labels:{{color:'#c9d1d9'}}}}}},
  scales:{{x:{{ticks:{{color:'#586069',maxTicksLimit:12}},grid:{{color:'#1c2730'}}}},
           y:{{ticks:{{color:'#586069',callback:v=>v+'%'}},grid:{{color:'#1c2730'}}}}}}}}
}});
</script>
</body>
</html>"""

        with open(output_path, "w") as f:
            f.write(html)
        print(f"[Trainer] Report → {output_path}")
        return output_path
