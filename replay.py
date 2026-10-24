"""
replay.py - walk through stored backtest predictions one match at a time. Offline and deterministic by default.

For each chosen match it shows:
  1. the match context and the exact prompt the model saw (built by backtest.build_prompt),
  2. the model's raw probability that team 1 wins (read from the DB; --live asks the model again),
  3. the Platt-calibrated probability, from a calibrator fitted on every OTHER match in the DB,
  4. (after a pause) the real result, the Brier score of the raw and the calibrated probability.
With --plot it also saves a reliability diagram (raw vs Platt, the Platt curve is out-of-fold).

Usage
  python replay.py --db backtest_deepseek.db                      # one random match, pauses before the reveal
  python replay.py --db backtest_deepseek.db --count 5 --no-pause # five matches, no pauses
  python replay.py --db backtest_deepseek.db --match-id 1234567   # a specific match
  python replay.py --db backtest_deepseek.db --plot reliability_replay.png --count 0   # only the plot
  python replay.py --db backtest_deepseek.db --live               # re-ask the model (needs DEEPSEEK_API_KEY; costs a little)
Needs backtest.py (or backtesting.py) in the same folder, for the prompt and the calibrator.
"""

import argparse
import sqlite3
import sys

import numpy as np
import pandas as pd

try:
    import backtest as bt
except ImportError:
    import backtesting as bt   # the name of the copy on my machine

EDGES = (0.2, 0.4, 0.6, 0.8)


def load(db_path: str, model: str = None) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT * FROM backtest_predictions WHERE brier_score IS NOT NULL ORDER BY id", conn)
    conn.close()
    if model:
        df = df[df["model"] == model]
    df = df[df["actual_outcome"].isin(["YES", "NO"])].copy()
    df["actual_numeric"] = (df["actual_outcome"] == "YES").astype(int)
    return df.reset_index(drop=True)


def bar(p: float, width: int = 40) -> str:
    n = int(round(p * width))
    return "[" + "#" * n + "." * (width - n) + f"] {p * 100:5.1f}%"


def replay_one(row, df, live: bool, pause: bool) -> dict:
    train = df[df["matchId"] != row["matchId"]]
    calibrator = bt._fit_platt(train)
    include_series = row.get("prompt_variant") != "no-series"
    info = {"team1": row["team1"], "team2": row["team2"], "matchFormat": row["match_format"],
            "seriesName": row["seriesName"], "venue": row["venue"]}

    print("=" * 74)
    print(f"{row['team1']} vs {row['team2']}   ({row['match_format']}, {str(row['match_date'])[:10]})")
    print("=" * 74)
    print("PROMPT SENT TO THE MODEL")
    print("-" * 74)
    print(bt.build_prompt(info, include_series=include_series).strip())
    print("-" * 74)

    raw = float(row["probability"])
    question = row["question"]
    if live:
        out = bt.generate_backtest_question(info)
        if out is None:
            print("Live call failed; using the stored answer.")
        else:
            raw, question = out["probability"], out["question"]

    one = pd.DataFrame({"probability": [raw]})
    cal = float(bt._apply_platt(calibrator, one)[0])
    print(f"MODEL ANSWER ({'live call' if live else 'stored'}): {question}")
    print(f"  raw        P({row['team1']} wins) {bar(raw)}")
    print(f"  calibrated P({row['team1']} wins) {bar(cal)}   (Platt, fitted on {len(train)} other matches)")
    shift = (cal - raw) * 100
    print(f"  calibration moved it {shift:+.1f} points")

    if pause:
        input("\nPress Enter to reveal the result... ")
    y = int(row["actual_numeric"])
    b_raw, b_cal = (raw - y) ** 2, (cal - y) ** 2
    print(f"\nRESULT: {row['team1']} {'WON' if y else 'did not win'} (team 2: {row['team2']})")
    print(f"  Brier raw        {b_raw:.4f}")
    print(f"  Brier calibrated {b_cal:.4f}   ({'better' if b_cal < b_raw else 'worse' if b_cal > b_raw else 'same'})")
    print("  (Brier = (probability - outcome)^2; 0 is perfect, always saying 50% scores 0.25)\n")
    return {"raw": raw, "cal": cal, "y": y, "b_raw": b_raw, "b_cal": b_cal}


def reliability_plot(df: pd.DataFrame, path: str, seed: int = 0) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.model_selection import KFold

    p = df["probability"].to_numpy(float)
    y = df["actual_numeric"].to_numpy(float)
    cal = np.zeros(len(df))
    for tr, te in KFold(5, shuffle=True, random_state=seed).split(df):   # out-of-fold, so the curve is not graded on its own training data
        c = bt._fit_platt(df.iloc[tr])
        cal[te] = bt._apply_platt(c, df.iloc[te])

    fig, ax = plt.subplots(figsize=(5.6, 5.6))
    ax.plot([0, 1], [0, 1], color="#999999", lw=1, ls="--", label="perfect calibration")
    for probs, name, color in ((p, "raw", "#d95f02"), (cal, "Platt (out-of-fold)", "#1b7837")):
        idx = np.digitize(probs, EDGES, right=True)
        xs, ys, ns = [], [], []
        for k in range(len(EDGES) + 1):
            m = idx == k
            if m.any():
                xs.append(probs[m].mean()); ys.append(y[m].mean()); ns.append(m.sum())
        e = bt._ece(probs, y) * 100
        ax.plot(xs, ys, "-", color=color, lw=1.2, alpha=0.7)
        ax.scatter(xs, ys, s=np.array(ns) / max(ns) * 160 + 20, color=color, label=f"{name}  (ECE {e:.1f})", zorder=3)
    ax.set_xlabel("predicted probability that team 1 wins")
    ax.set_ylabel("share of matches team 1 actually won")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title(f"Reliability, {len(df)} matches (dot size = matches in bin)", fontsize=10)
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    print(f"Saved reliability diagram to {path}")


def main():
    ap = argparse.ArgumentParser(description="Replay stored backtest predictions.")
    ap.add_argument("--db", default=bt.DEFAULT_DB)
    ap.add_argument("--model", default=None, help="only rows made by this model")
    ap.add_argument("--match-id", default=None)
    ap.add_argument("--count", type=int, default=1, help="how many random matches to replay (0 = none, e.g. for --plot only)")
    ap.add_argument("--seed", type=int, default=None, help="fix the random draw so the demo is repeatable")
    ap.add_argument("--no-pause", action="store_true")
    ap.add_argument("--live", action="store_true", help="ask the model again instead of reading the stored probability")
    ap.add_argument("--plot", default=None, help="save a reliability diagram here (e.g. reliability_replay.png)")
    args = ap.parse_args()

    df = load(args.db, args.model)
    if len(df) < 30:
        sys.exit(f"Only {len(df)} scored rows in {args.db}; need at least 30 to fit a calibrator.")
    print(f"{len(df)} scored matches in {args.db}\n")

    if args.match_id:
        rows = df[df["matchId"].astype(str) == str(args.match_id)]
        if rows.empty:
            sys.exit(f"match {args.match_id} not found")
    else:
        rows = df.sample(n=min(args.count, len(df)), random_state=args.seed) if args.count > 0 else df.iloc[0:0]

    results = [replay_one(r, df, args.live, not args.no_pause) for _, r in rows.iterrows()]
    if len(results) > 1:
        r = pd.DataFrame(results)
        print(f"Over these {len(r)} matches: Brier raw {r.b_raw.mean():.4f}, calibrated {r.b_cal.mean():.4f}  (small sample, only illustrative)")
    if args.plot:
        reliability_plot(df, args.plot)


if __name__ == "__main__":
    main()