#!/usr/bin/env python3
"""
xgb_baseline.py - a feature-based baseline for the cricket backtest.

Question it answers: how does an XGBoost model built from pre-match features
compare with the prompt-only LLM, on the SAME matches the LLM was scored on?

How it stays fair
  * Features for a match use only matches played on EARLIER dates (Elo ratings,
    recent form, head-to-head, venue record). Nothing from the match itself.
  * Walk-forward: for evaluation matches in year Y the model is trained only on
    matches dated before 1 January of Y. No future leakage, no tuning on the
    evaluation matches (hyper-parameters are fixed below).
  * Toss is left out by default (the live engine generates hours before the toss).
  * Each match is predicted in both team orders and averaged, so the arbitrary
    "team 1" label cannot matter.
  * The LLM's stored probabilities are read from the backtest database; no API calls.

No LLM contamination issue applies to XGBoost (it has no pretrained memory), so it
is trained on every date. Use --after YYYY-MM-DD to REPORT only on matches after a
cut-off, which is the like-for-like check against a possibly memorising LLM.

Usage
  python xgb_baseline.py --data-dir /path/to/cricsheet_json --db backtest_deepseek.db
  python xgb_baseline.py --data-dir ... --db ... --after 2025-01-01
  python xgb_baseline.py --data-dir ... --db ... --with-toss        # reference only
The first run parses every Cricsheet file (a few minutes) and caches the result in
cricsheet_meta.csv; later runs start in seconds.
"""
import argparse
import json
import math
import os
import re
import sqlite3
import sys
from collections import defaultdict, deque
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

HIGH_ENGAGEMENT_TEAMS = ["India", "Australia", "England", "Pakistan",
                         "South Africa", "New Zealand", "West Indies", "Bangladesh"]
FMT_GROUP = {"Test": 0, "MDM": 0, "ODI": 1, "ODM": 1, "T20": 2, "IT20": 2}
ELO_K = 20.0
EDGES = (0.2, 0.4, 0.6, 0.8)
MIN_BIN = 30
DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")


# ---------------------------------------------------------------------------
# 1. Parse the Cricsheet archive (cached)
# ---------------------------------------------------------------------------

def _parse_meta(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
    except Exception:
        return None
    info = d.get("info", {})
    teams = info.get("teams", [])
    dates = info.get("dates") or []
    if len(teams) < 2 or not dates:
        return None
    outcome = info.get("outcome", {}) or {}
    toss = info.get("toss", {}) or {}
    innings = d.get("innings") or []
    return {
        "matchId": os.path.basename(path)[:-5],
        "date": str(dates[0]),
        "fmt": info.get("match_type", "Unknown"),
        "team_type": info.get("team_type", ""),
        "gender": info.get("gender", ""),
        "team1": teams[0], "team2": teams[1],
        "venue": info.get("venue", "Unknown"),
        "winner": outcome.get("winner"),
        "toss_winner": toss.get("winner"),
        "toss_decision": toss.get("decision"),
        "first_bat": innings[0].get("team") if innings else None,
    }


def load_meta(data_dir, cache, workers):
    if cache and os.path.exists(cache):
        print(f"Reading cached match table {cache}")
        return pd.read_csv(cache, dtype={"matchId": str})
    files = sorted(os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".json"))
    print(f"Parsing {len(files)} Cricsheet files with {workers} worker(s)...")
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, r in enumerate(ex.map(_parse_meta, files, chunksize=64), 1):
            if r:
                rows.append(r)
            if i % 5000 == 0:
                print(f"  {i}/{len(files)}")
    meta = pd.DataFrame(rows)
    if cache:
        meta.to_csv(cache, index=False)
        print(f"Saved {cache}")
    return meta


# ---------------------------------------------------------------------------
# 2. Walk-forward features: every row uses only EARLIER dates
# ---------------------------------------------------------------------------

FEATURES = ["elo_diff", "felo_diff", "elo_mean", "form5_diff", "form10_diff",
            "exp_a", "exp_b", "rest_a", "rest_b", "venue_rate_diff", "venue_games",
            "h2h_rate", "h2h_games", "bf_rate", "bf_games", "fmt", "intl"]
TOSS_FEATURES = ["toss_a", "toss_bat"]


def build_features(meta, with_toss=False):
    """Returns (meta_sorted, X1, X2, feature_names). X1 is team1's view, X2 team2's view."""
    meta = meta[meta["date"].astype(str).str.fullmatch(DATE_RE.pattern)].copy()
    meta["date"] = meta["date"].astype(str)
    meta = meta.sort_values(["date", "matchId"]).reset_index(drop=True)
    meta["ord"] = pd.to_datetime(meta["date"]).map(pd.Timestamp.toordinal)
    meta["year"] = meta["date"].str[:4].astype(int)

    elo = defaultdict(lambda: 1500.0)
    felo = defaultdict(lambda: 1500.0)
    recent = defaultdict(lambda: deque(maxlen=10))
    played = defaultdict(int)
    last = {}
    h2h = defaultdict(lambda: [0, 0])      # (a, b) -> [a's wins, games]
    vt = defaultdict(lambda: [0, 0])       # (venue, team) -> [wins, games]
    vbf = defaultdict(lambda: [0, 0])      # venue -> [bat-first wins, games]

    names = FEATURES + (TOSS_FEATURES if with_toss else [])
    X1 = np.zeros((len(meta), len(names)))
    X2 = np.zeros((len(meta), len(names)))

    def side(team, fg, o, venue):
        rec = list(recent[team])
        k5 = rec[-5:]
        vw, vg = vt[(venue, team)]
        return {
            "elo": elo[team], "felo": felo[(team, fg)],
            "f5": (sum(k5) + 1.5) / (len(k5) + 3), "f10": (sum(rec) + 3) / (len(rec) + 6),
            "exp": math.log1p(played[team]),
            "rest": min(o - last[team], 90) if team in last else 90,
            "vrate": (vw + 1) / (vg + 2), "vg": vg,
        }

    def row(a, b, sa, sb, fg, venue, intl, toss_w, toss_d):
        hw, hg = h2h[(a, b)]
        bw, bg = vbf[venue]
        r = [sa["elo"] - sb["elo"], sa["felo"] - sb["felo"], (sa["elo"] + sb["elo"]) / 2,
             sa["f5"] - sb["f5"], sa["f10"] - sb["f10"],
             sa["exp"], sb["exp"], sa["rest"], sb["rest"],
             sa["vrate"] - sb["vrate"], sa["vg"] + sb["vg"],
             (hw + 1) / (hg + 2), hg, (bw + 1) / (bg + 2), bg, fg, intl]
        if with_toss:
            r += [1.0 if toss_w == a else 0.0, 1.0 if toss_d == "bat" else 0.0]
        return r

    i0 = 0
    for date, g in meta.groupby("date", sort=True):
        idxs = list(g.index)
        for i in idxs:                      # features first, from the state BEFORE this date
            m = meta.loc[i]
            a, b, fg = m["team1"], m["team2"], FMT_GROUP.get(m["fmt"], 1)
            o, venue = int(m["ord"]), m["venue"]
            intl = 1.0 if m["team_type"] == "international" else 0.0
            sa, sb = side(a, fg, o, venue), side(b, fg, o, venue)
            X1[i] = row(a, b, sa, sb, fg, venue, intl, m["toss_winner"], m["toss_decision"])
            X2[i] = row(b, a, sb, sa, fg, venue, intl, m["toss_winner"], m["toss_decision"])
        for i in idxs:                      # then update the state with this date's results
            m = meta.loc[i]
            a, b, w = m["team1"], m["team2"], m["winner"]
            if w not in (a, b):
                continue                    # tie / no result: nothing to learn from
            fg, o, venue = FMT_GROUP.get(m["fmt"], 1), int(m["ord"]), m["venue"]
            sa_ = 1.0 if w == a else 0.0
            ea = 1.0 / (1.0 + 10 ** ((elo[b] - elo[a]) / 400.0))
            elo[a] += ELO_K * (sa_ - ea); elo[b] -= ELO_K * (sa_ - ea)
            fa = 1.0 / (1.0 + 10 ** ((felo[(b, fg)] - felo[(a, fg)]) / 400.0))
            felo[(a, fg)] += ELO_K * (sa_ - fa); felo[(b, fg)] -= ELO_K * (sa_ - fa)
            recent[a].append(sa_); recent[b].append(1 - sa_)
            played[a] += 1; played[b] += 1
            last[a] = o; last[b] = o
            h2h[(a, b)][0] += int(sa_); h2h[(a, b)][1] += 1
            h2h[(b, a)][0] += int(1 - sa_); h2h[(b, a)][1] += 1
            vt[(venue, a)][0] += int(sa_); vt[(venue, a)][1] += 1
            vt[(venue, b)][0] += int(1 - sa_); vt[(venue, b)][1] += 1
            fb = m["first_bat"]
            if isinstance(fb, str) and fb in (a, b):
                vbf[venue][1] += 1
                vbf[venue][0] += int(w == fb)
    meta["y"] = np.where(meta["winner"] == meta["team1"], 1.0,
                         np.where(meta["winner"] == meta["team2"], 0.0, np.nan))
    return meta, X1, X2, names


# ---------------------------------------------------------------------------
# 3. Models and the walk-forward loop
# ---------------------------------------------------------------------------

class EloLogit:
    """Logistic regression on the two Elo differences only: the 'no machine learning' yardstick."""
    def __init__(self, cols):
        self.cols = cols

    def fit(self, X, y):
        from sklearn.linear_model import LogisticRegression
        self.m = LogisticRegression(C=1.0, max_iter=1000).fit(X[:, self.cols] / 100.0, y)
        return self

    def predict_proba(self, X):
        return self.m.predict_proba(X[:, self.cols] / 100.0)


def make_xgb(seed):
    try:
        from xgboost import XGBClassifier
    except ImportError:
        sys.exit("xgboost is not installed: pip install xgboost")
    # Fixed, deliberately small: nothing here is tuned on the evaluation matches.
    return XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.03, subsample=0.8,
                         colsample_bytree=0.8, min_child_weight=5, reg_lambda=5.0,
                         eval_metric="logloss", tree_method="hist", n_jobs=2, random_state=seed)


def walk_forward(meta, X1, X2, elig, eval_mask, factory, min_train):
    """Predict each evaluation match with a model trained only on eligible matches dated
    before 1 Jan of its year. Both team orders are averaged. Returns (preds, {year: n_train}, last_model)."""
    preds = np.full(len(meta), np.nan)
    sizes, model = {}, None
    for year in sorted(meta.loc[eval_mask, "year"].unique()):
        tr = elig & (meta["date"] < f"{year}-01-01").values
        if tr.sum() < min_train:
            continue
        Xtr = np.vstack([X1[tr], X2[tr]])
        ytr = np.concatenate([meta.loc[tr, "y"].values, 1.0 - meta.loc[tr, "y"].values])
        model = factory().fit(Xtr, ytr)
        te = eval_mask & (meta["year"] == year).values
        p1 = model.predict_proba(X1[te])[:, 1]
        p2 = model.predict_proba(X2[te])[:, 1]
        preds[te] = (p1 + 1.0 - p2) / 2.0
        sizes[int(year)] = int(tr.sum())
    return preds, sizes, model


# ---------------------------------------------------------------------------
# 4. Metrics (self-contained; same definitions as the backtest report)
# ---------------------------------------------------------------------------

def brier(p, y):
    return float(np.mean((p - y) ** 2))


def acc(p, y):
    return float(np.mean((p >= 0.5) == (y == 1)))


def _bins(p, y):
    idx = np.digitize(p, EDGES)
    out = []
    for k in range(5):
        m = idx == k
        n = int(m.sum())
        out.append((n, float(p[m].mean()) if n else np.nan, float(y[m].mean()) if n else np.nan))
    return out


def ece(p, y):
    n_all = len(p)
    return float(sum(n / n_all * abs(pm - ym) for n, pm, ym in _bins(p, y) if n)) * 100


def worst_gap(p, y):
    gaps = [abs(ym - pm) for n, pm, ym in _bins(p, y) if n >= MIN_BIN]
    return float(max(gaps)) * 100 if gaps else float("nan")


def ece_floor(p, sims=500, seed=0):
    rng = np.random.default_rng(seed)
    draws = np.array([ece(p, (rng.random(len(p)) < p).astype(float)) for _ in range(sims)])
    return float(draws.mean()), float(np.percentile(draws, 95))


def boot_ci(v, reps=2000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(v)
    means = np.array([v[rng.integers(0, n, n)].mean() for _ in range(reps)])
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def blend_oof(p_llm, p_x, y, repeats=20, folds=5, seed=0):
    """Stack the two probabilities with a 2-feature logistic regression, fully out of fold."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    Z = np.column_stack([logit(p_llm), logit(p_x)])
    b, a, e, w = [], [], [], []
    for r in range(repeats):
        oof = np.zeros(len(y))
        for tr, te in StratifiedKFold(folds, shuffle=True, random_state=seed + r).split(Z, y):
            oof[te] = LogisticRegression(C=1.0, max_iter=1000).fit(Z[tr], y[tr]).predict_proba(Z[te])[:, 1]
        b.append(brier(oof, y)); a.append(acc(oof, y)); e.append(ece(oof, y)); w.append(worst_gap(oof, y))
    full = LogisticRegression(C=1.0, max_iter=1000).fit(Z, y)
    return np.mean(b), np.mean(a), np.mean(e), np.mean(w), full.coef_[0], full.intercept_[0]


# ---------------------------------------------------------------------------
# 5. Main
# ---------------------------------------------------------------------------

def load_llm(db, model):
    conn = sqlite3.connect(db)
    q = ("SELECT matchId, team1, probability AS p_llm, actual_outcome FROM backtest_predictions "
         "WHERE brier_score IS NOT NULL")
    params = ()
    if model:
        q += " AND model = ?"
        params = (model,)
    df = pd.read_sql_query(q, conn, params=params)
    conn.close()
    df["matchId"] = df["matchId"].astype(str)
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True, help="folder of Cricsheet JSON files")
    ap.add_argument("--db", default="backtest_deepseek.db", help="backtest database with the LLM's probabilities")
    ap.add_argument("--model", default=None, help="only LLM rows from this model name")
    ap.add_argument("--cache", default="cricsheet_meta.csv", help="parsed match table; delete it to re-parse")
    ap.add_argument("--after", default=None, help="report only matches dated on/after YYYY-MM-DD")
    ap.add_argument("--with-toss", action="store_true", help="add toss features (reference only; unknown at prediction time)")
    ap.add_argument("--train-scope", choices=["engaged", "international", "all"], default="engaged",
                    help="which matches XGBoost trains on (default: those with a high-engagement team, like the backtest)")
    ap.add_argument("--min-train", type=int, default=300, help="skip years with fewer training matches")
    ap.add_argument("--workers", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save", default="", help="optional CSV of per-match predictions")
    args = ap.parse_args()

    meta = load_meta(args.data_dir, args.cache, args.workers)
    meta, X1, X2, names = build_features(meta, args.with_toss)
    decisive = meta["y"].notna().values
    engaged = (meta["team1"].isin(HIGH_ENGAGEMENT_TEAMS) | meta["team2"].isin(HIGH_ENGAGEMENT_TEAMS)).values
    intl = (meta["team_type"] == "international").values
    elig = decisive & {"engaged": engaged, "international": intl, "all": np.ones(len(meta), bool)}[args.train_scope]
    print(f"{len(meta)} matches parsed; {int(elig.sum())} usable for training "
          f"(scope '{args.train_scope}', decisive results only).")

    llm = load_llm(args.db, args.model)
    llm = llm.merge(meta[["matchId", "team1", "y", "date"]].rename(columns={"team1": "meta_team1"}),
                    on="matchId", how="inner")
    flipped = llm["team1"] != llm["meta_team1"]
    if flipped.any():                        # keep everything in the database's team-1 orientation
        llm.loc[flipped, "p_llm"] = 1.0 - llm.loc[flipped, "p_llm"]
        print(f"Note: {int(flipped.sum())} rows had team order different from the Cricsheet file; re-oriented.")
    y_db = (llm["actual_outcome"] == "YES").astype(float)
    y_meta = np.where(flipped, 1.0 - llm["y"], llm["y"])
    bad = int((y_db != y_meta).sum())
    print(f"LLM rows matched to Cricsheet files: {len(llm)}; outcome disagreements with the files: {bad}")
    if args.after:
        llm = llm[llm["date"] >= args.after]
        y_db = y_db[llm.index]
        print(f"Reporting only matches dated {args.after} or later: {len(llm)} matches.")

    pos = pd.Series(np.arange(len(meta)), index=meta["matchId"])
    eval_mask = np.zeros(len(meta), bool)
    eval_mask[pos[llm["matchId"]].values] = True
    # In the database's orientation the walk-forward model predicts team1 of the Cricsheet file.
    xgb_p, sizes, last_model = walk_forward(meta, X1, X2, elig, eval_mask, lambda: make_xgb(args.seed), args.min_train)
    elo_cols = [names.index("elo_diff"), names.index("felo_diff")]
    elo_p, _, _ = walk_forward(meta, X1, X2, elig, eval_mask, lambda: EloLogit(elo_cols), args.min_train)

    idx = pos[llm["matchId"]].values
    d = pd.DataFrame({"matchId": llm["matchId"].values, "date": llm["date"].values,
                      "y": y_db.values if hasattr(y_db, "values") else np.asarray(y_db),
                      "p_llm": llm["p_llm"].values,
                      "p_xgb": np.where(flipped.loc[llm.index].values, 1 - xgb_p[idx], xgb_p[idx]),
                      "p_elo": np.where(flipped.loc[llm.index].values, 1 - elo_p[idx], elo_p[idx])})
    d["year"] = d["date"].str[:4].astype(int)
    n_all = len(d)
    d = d.dropna(subset=["p_xgb", "p_elo"]).reset_index(drop=True)
    print(f"\nMatches with a walk-forward prediction: {len(d)} of {n_all} "
          f"(the rest are from years with fewer than {args.min_train} earlier training matches).")
    if len(d) < 100:
        sys.exit("Too few matches to compare; check --data-dir, --db and --min-train.")
    y = d["y"].values

    print("\n" + "=" * 74)
    print(f" LLM vs feature-based models on the SAME {len(d)} matches"
          + (f" (dated {args.after} or later)" if args.after else ""))
    print("=" * 74)
    print(f"base rate of team 1 winning: {y.mean():.1%}   (constant-rate Brier {y.mean() * (1 - y.mean()):.4f})")
    fl_mean, fl_95 = ece_floor(d["p_llm"].values, seed=args.seed)
    print(f"ECE from sampling noise alone at this size (LLM probabilities): {fl_mean:.1f} pts (95th pct {fl_95:.1f})\n")
    print(f"{'model':<34}{'Brier (95% CI)':<28}{'accuracy':>9}{'ECE':>7}{'worst gap':>11}")
    rows = [("LLM (DeepSeek, prompt only)", d["p_llm"].values),
            ("Elo only (logistic)", d["p_elo"].values),
            ("XGBoost" + (" + toss" if args.with_toss else ""), d["p_xgb"].values)]
    for name, p in rows:
        lo, hi = boot_ci((p - y) ** 2, seed=args.seed)
        print(f"{name:<34}{brier(p, y):.4f} [{lo:.4f}, {hi:.4f}]   {acc(p, y):>8.1%}{ece(p, y):>7.1f}{worst_gap(p, y):>11.1f}")
    bb, ba, be, bw, coef, icpt = blend_oof(d["p_llm"].values, d["p_xgb"].values, y, seed=args.seed)
    print(f"{'LLM + XGBoost blend (5-fold OOF)':<34}{bb:.4f}{'':<19}   {ba:>8.1%}{be:>7.1f}{bw:>11.1f}")
    print(f"  blend weights on log-odds: LLM {coef[0]:+.2f}, XGBoost {coef[1]:+.2f}, intercept {icpt:+.2f} "
          "(descriptive; both positive = each adds information the other lacks)")

    print("\nPaired differences on the same matches (bootstrap 95% interval):")
    for a_name, pa, b_name, pb in [("XGBoost", d["p_xgb"].values, "LLM", d["p_llm"].values),
                                   ("Elo only", d["p_elo"].values, "LLM", d["p_llm"].values),
                                   ("XGBoost", d["p_xgb"].values, "Elo only", d["p_elo"].values)]:
        diff = (pa - y) ** 2 - (pb - y) ** 2
        lo, hi = boot_ci(diff, seed=args.seed)
        verdict = "no clear difference" if lo <= 0 <= hi else (f"{a_name} better" if hi < 0 else f"{b_name} better")
        print(f"  Brier {a_name} - {b_name}: {diff.mean():+.4f}  [{lo:+.4f}, {hi:+.4f}]  ({verdict}; negative = {a_name} better)")
        adiff = ((pa >= 0.5) == (y == 1)).astype(float) - ((pb >= 0.5) == (y == 1)).astype(float)
        lo, hi = boot_ci(adiff, seed=args.seed)
        print(f"  accuracy {a_name} - {b_name}: {adiff.mean() * 100:+.1f} pts  [{lo * 100:+.1f}, {hi * 100:+.1f}]")

    print("\nBy year (does more training data help?):")
    print(f"{'year':<6}{'train n':>9}{'eval n':>8}{'LLM Brier':>11}{'XGB Brier':>11}")
    for yr, g in d.groupby("year"):
        print(f"{yr:<6}{sizes.get(int(yr), 0):>9}{len(g):>8}{brier(g['p_llm'].values, g['y'].values):>11.4f}"
              f"{brier(g['p_xgb'].values, g['y'].values):>11.4f}")

    if last_model is not None and hasattr(last_model, "feature_importances_"):
        imp = sorted(zip(names, last_model.feature_importances_), key=lambda t: -t[1])[:8]
        print("\nXGBoost feature importance (last year's model, descriptive only): "
              + ", ".join(f"{n} {v:.2f}" for n, v in imp))

    if args.save:
        d.to_csv(args.save, index=False)
        print(f"Saved per-match predictions to {args.save}")


if __name__ == "__main__":
    main()