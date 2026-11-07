"""Like-for-like check of the raw 20-40% bin gap, plus the in-sample ECE of the Platt fit.

Run from the folder that holds backtest.py (or backtesting.py):
    python check_bin_gap.py --db backtest_deepseek.db
Add --model <name> if the DB holds more than one model.

It follows the SAME matches through calibration: take the matches whose RAW probability is in a bin, apply the
calibrator fitted on all matches, and compare their mean calibrated probability with their actual win rate.
(The doc's calibrated-bin table re-sorts matches by the calibrated probability, so its bins hold different matches.)
"""
import argparse

import numpy as np

try:
    import backtest as bt
except ImportError:
    import backtesting as bt

ap = argparse.ArgumentParser()
ap.add_argument("--db", default="backtest_deepseek.db")
ap.add_argument("--model", default=None)
args = ap.parse_args()

df = bt.load_scored(args.db, args.model)
df = df[df["actual_outcome"].isin(["YES", "NO"])]          # VOID matches are not scored
p = df["probability"].to_numpy(float)
y = df["actual_numeric"].to_numpy(float)
print(f"{len(p)} scored matches\n")

fits = {m: bt._fit_calibrator(m, p, y) for m in ("platt", "isotonic")}
cal = {m: bt._apply_calibrator(m, fits[m], p) for m in fits}

c = fits["platt"]
print(f"Platt fit on all matches: slope {c.coef_[0][0]:.3f}, intercept {c.intercept_[0]:.3f} "
      f"(sklearn's LogisticRegression default is lightly regularised)")
print(f"ECE, raw:                       {bt._ece(p, y) * 100:.2f} pts")
for m in fits:
    print(f"ECE, {m:8s} fitted on all, scored on all (in-sample): {bt._ece(cal[m], y) * 100:.2f} pts")

print("\nSAME matches followed through calibration (bins by RAW probability; edges are (0.2,0.4] etc, as in the doc):")
print(f"{'raw bin':9s} {'n':>5s} {'mean raw':>9s} {'actual':>8s} {'raw gap':>8s} | "
      f"{'Platt mean':>10s} {'gap':>6s} | {'isotonic mean':>13s} {'gap':>6s}")
idx = np.digitize(p, bt.EDGES, right=True)
for k, label in enumerate(bt.BIN_LABELS):
    m = idx == k
    if not m.any():
        continue
    act = y[m].mean() * 100
    row = f"{label:9s} {m.sum():5d} {p[m].mean() * 100:9.1f} {act:8.1f} {act - p[m].mean() * 100:+8.1f} | "
    for name in ("platt", "isotonic"):
        mc = cal[name][m].mean() * 100
        row += f"{mc:10.1f} {act - mc:+6.1f} | " if name == "platt" else f"{mc:13.1f} {act - mc:+6.1f}"
    print(row)
print("\ngap = actual win rate minus mean prediction, in points. The 20-40% row is the one in question.")