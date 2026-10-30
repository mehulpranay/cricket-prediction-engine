"""
Cricket backtest on the Cricsheet archive — DeepSeek version.

For each sampled historical match the model sees ONLY pre-match context (teams, format, venue, series name; see
INCLUDE_SERIES below), returns
P(team1 wins), and is scored against the real result (Brier + directional accuracy). Then: calibration table,
Platt scaling (60/40 split) and an expanding-window calibration test.

Compared with the notebook:
  * runs as a script (Kaggle or local), LLM = DeepSeek (OpenAI-compatible API)
  * every row stores which model produced it, and results go to a NEW file (backtest_deepseek.db) so they are
    never mixed with the old gpt-4o-mini rows
  * a match already in the DB is never scored twice (the notebook could append repeats), and one fixed shuffle
    (--seed) is walked through across runs
  * the report adds 95% bootstrap intervals, the base-rate baseline, a per-format table, and a plain
    "improved in X of N chunks" summary for the expanding window

Position-bias check (does the model favour whichever team is NAMED FIRST?): the prompt always names team1 first
and asks about team1. Scoring a match in BOTH orders, and comparing, measures that bias and removes most of it:
    python backtest.py --fill-swaps --n 250      # add the swapped-order probability to matches already in the DB (1 call each)
    python backtest.py --swap-teams --n 250      # score NEW matches in both orders (2 calls each)
The report then adds a "POSITION-BIAS CHECK" section. (--fill-swaps only touches rows made by the current model.)

Calibration experiment (no API calls; uses the predictions already stored in the DB): is recalibration worth it,
and does isotonic regression need more data?
    python backtest.py --eval-only --calibration-experiment            # raw vs Platt vs isotonic, out-of-fold
    python backtest.py --eval-only --calibration-experiment --prob-source avg    # use the order-averaged probability
    python backtest.py --calibration-only                              # JUST the experiment + learning curve (no long report, no API calls)
Calibration (ECE, and the gap in each confidence bin) is the target; Brier is only a no-harm check, because recalibration
can move Brier by at most its small reliability term. The experiment reports, out-of-fold (every match is scored by a
calibrator that never saw it): ECE for raw / Platt / isotonic, the plain reduction vs raw (raw ECE minus out-of-fold ECE)
with an approximate 95% interval (a bootstrap re-centred on that point estimate, because the plain bootstrap understates
the reduction), a noise-only test of whether the RAW ECE is real, the worst per-bin gap, the per-bin gap table, the ECE
you would see from sampling noise alone, Platt's fitted slope, and an ECE LEARNING CURVE (ECE vs calibration training
size). The expanding-window test reports CUMULATIVE out-of-sample ECE, and the calibration table carries a Wilson 95%
interval per bin ('*' = the gap is larger than sampling noise).

Setup:  DEEPSEEK_API_KEY in the environment; pip install openai numpy pandas scikit-learn matplotlib
Run:
    python backtest.py --data-dir /path/to/cricsheet_json --n 250     # score up to 250 NEW matches, then report
    python backtest.py --eval-only                                    # report from the DB, no API calls

Series name: the prompt carries the Cricsheet series/event name (e.g. "ICC Cricket World Cup", "India tour of Australia"),
which is real pre-match context: it tells the model who is at home and how much the match matters. It can also hint at
the exact edition (a year, a sponsor), so as an ablation you can re-score the SAME matches without it and measure what
it was worth:
    python backtest.py --db backtest_deepseek_noseries.db --match-ids-from backtest_deepseek.db --no-series-name --n 2000
    python backtest.py --eval-only --db backtest_deepseek_noseries.db --compare-with backtest_deepseek.db
A DB remembers which prompt made its rows and refuses to mix the two.

Contamination control: --after YYYY-MM-DD scores only matches played on or after that date. Set it to the model's
training cutoff (plus a safety margin; cutoffs are often reported inconsistently) so the model cannot have seen the
result. Use a SEPARATE --db for such a run, so the filtered and unfiltered runs are never mixed in one report.
    python backtest.py --after 2025-01-01 --db backtest_deepseek_postcutoff.db --n 250
"""

import argparse
import json
import math
import os
import random
import re
import sqlite3
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from openai import OpenAI

DEEPSEEK_BASE_URL = "https://api.deepseek.com"
MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-flash")   # verify against DeepSeek's current model list
DEFAULT_DB = "backtest_deepseek.db"
DEFAULT_DATA_DIR = "/kaggle/input/datasets/mehulkumar99/cricksheet-data-22000-matches"

# High-engagement teams list (format agnostic)
HIGH_ENGAGEMENT_TEAMS = [
    "India", "Australia", "England", "Pakistan",
    "South Africa", "New Zealand", "West Indies", "Bangladesh"
]

# The prompt includes the Cricsheet series/event name (real context: World Cup, home/away tour). --no-series-name drops it,
# for the ablation that measures how much it contributes (use a separate --db).
INCLUDE_SERIES = True

_client = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url=DEEPSEEK_BASE_URL)
    return _client


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------

def init_backtest_db(db_path: str) -> None:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS backtest_predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            matchId TEXT,
            team1 TEXT,
            team2 TEXT,
            seriesName TEXT,
            match_format TEXT,
            match_date TEXT,
            venue TEXT,
            question TEXT,
            probability REAL,
            actual_outcome TEXT,
            is_correct INTEGER,
            brier_score REAL,
            created_at TEXT,
            model TEXT,
            prob_team2_first REAL,
            prompt_variant TEXT
        )
    ''')
    existing = {row[1] for row in cursor.execute("PRAGMA table_info(backtest_predictions)")}
    for column, decl in (("model", "TEXT"), ("prob_team2_first", "REAL"), ("prompt_variant", "TEXT")):   # older DB files lack these
        if column not in existing:
            cursor.execute(f"ALTER TABLE backtest_predictions ADD COLUMN {column} {decl}")
    conn.commit()
    conn.close()


def _variant() -> str:
    return "series" if INCLUDE_SERIES else "no-series"


def _check_variant(db_path: str) -> None:
    """Refuse to add rows to a DB made with a different prompt (rows with no label predate the flag and used the series name)."""
    conn = sqlite3.connect(db_path)
    have = {(r[0] or "series") for r in conn.execute("SELECT DISTINCT prompt_variant FROM backtest_predictions")}
    conn.close()
    other = have - {_variant()}
    if other:
        raise SystemExit(f"'{db_path}' already holds rows made with the '{', '.join(sorted(other))}' prompt; this run uses "
                         f"'{_variant()}'. Use a new --db so the two are never mixed.")


# ---------------------------------------------------------------------------
# Parsing + question generation
# ---------------------------------------------------------------------------

def parse_cricsheet_json(file_path): # parsing info of one match from cricsheet json file
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    info = data.get('info', {})
    teams = info.get('teams', [])
    if len(teams) < 2:
        return None

    team1, team2 = teams[0], teams[1]

    # Format-agnostic filter: At least one team must be a high-engagement team
    if team1 not in HIGH_ENGAGEMENT_TEAMS and team2 not in HIGH_ENGAGEMENT_TEAMS:
        return None

    match_type = info.get('match_type', 'Unknown')
    dates = info.get('dates', ['Unknown'])
    venue = info.get('venue', 'Unknown')
    event = info.get('event', {})
    series_name = event.get('name', f"{match_type} Match")

    outcome = info.get('outcome', {})
    winner = outcome.get('winner')  # Can be None if match was tied/drawn/no result

    match_id = os.path.basename(file_path).replace('.json', '')

    return {
        "matchId": match_id,
        "team1": team1,
        "team2": team2,
        "seriesName": series_name,
        "matchFormat": match_type,
        "matchDate": dates[0] if dates else "Unknown",
        "venue": venue,
        "winner": winner
    }


def _parse_prediction(content: str):
    """Validated {'question', 'probability'} from the model's text, or None."""
    content = (content or "").replace("```json", "").replace("```", "").strip()
    if not content:
        return None
    try:
        data = json.loads(content)
        probability = float(data["probability"])
        question = str(data.get("question") or "").strip()
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None
    if not (0.0 <= probability <= 1.0):
        return None
    return {"question": question, "probability": min(max(probability, 0.01), 0.99)}


def build_prompt(match_info, include_series=None):
    """The exact prompt sent to the model (replay.py imports this to show it)."""
    team1 = match_info['team1']
    team2 = match_info['team2']
    match_format = match_info['matchFormat']
    series_name = match_info['seriesName']
    venue = match_info['venue']
    use_series = INCLUDE_SERIES if include_series is None else include_series

    # The notebook's prompt. With --no-series-name the series line becomes a format-only line (the ablation).
    context_line = f"- Series: {series_name} ({match_format})" if use_series else f"- Format: {match_format}"
    return f"""
You are an expert cricket analyst and prediction market maker. 
Analyze the historical match context and output a JSON object with your market prediction.

Match Context:
{context_line}
- Teams: {team1} vs {team2}
- Venue: {venue}

You must output a JSON object containing EXACTLY these keys:
- "question": A binary question framed as "Will {team1} win this match against {team2}?"
- "probability": A float between 0.01 and 0.99 representing the estimated probability that {team1} wins.

"""


def generate_backtest_question(match_info):
    team1 = match_info['team1']
    team2 = match_info['team2']
    prompt = build_prompt(match_info)
    for attempt in (1, 2):   # JSON mode occasionally returns empty or malformed content; one retry is enough
        try:
            response = get_client().chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,   # zero temperature: repeatable predictions
                reasoning_effort="none",
                response_format={"type": "json_object"},
            )
            parsed = _parse_prediction(response.choices[0].message.content)
        except Exception as e:
            print(f"Error generating question for {team1} vs {team2} (attempt {attempt}): {e}")
            continue
        if parsed:
            return parsed
    return None


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def run_backtest_simulation(data_dir: str, db_path: str, n: int = 250, seed: int = 42, after: str = None,
                            swap_teams: bool = False, ids_from: str = None) -> int: 
    """Scores up to `n` NEW matches (never one already in the DB). Returns how many were added.
    With `after` ('YYYY-MM-DD'), only matches played on or after that date are considered.
    With `swap_teams`, each match is also scored with the two teams in swapped order (position-bias check).
    With `ids_from` (another backtest DB), scores exactly the matches that DB scored, in its order, instead of a random draw;
    use it with a NEW --db to re-score the same matches under a different prompt (paired comparison)."""
    if not os.path.isdir(data_dir):
        print(f"Directory '{data_dir}' not found. Pass --data-dir with the folder of Cricsheet JSON files.")
        return 0

    _check_variant(db_path)
    if ids_from: # re-score exactly the matches already scored in another DB, in its order
        src = sqlite3.connect(ids_from)
        old_ids = [r[0] for r in src.execute("SELECT matchId FROM backtest_predictions WHERE brier_score IS NOT NULL ORDER BY id")]
        src.close()
        json_files = [f"{m}.json" for m in old_ids if os.path.exists(os.path.join(data_dir, f"{m}.json"))]
        print(f"Re-scoring the {len(json_files)} matches stored in {ids_from} (of {len(old_ids)}; the rest have no file in --data-dir).")
    else:
        json_files = sorted(f for f in os.listdir(data_dir) if f.endswith('.json'))
        random.Random(seed).shuffle(json_files)   # one fixed shuffle; repeated runs continue down it
    print(f"Found {len(json_files)} files. Filtering and simulating (model: {MODEL}, prompt: {_variant()})...")
    if after:
        print(f"Contamination control: only matches dated {after} or later are scored.")

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    already_scored = {row[0] for row in cursor.execute("SELECT matchId FROM backtest_predictions")}

    processed_count = 0
    for fname in json_files:
        if processed_count >= n:
            break
        if fname[:-5] in already_scored:
            continue

        match_info = parse_cricsheet_json(os.path.join(data_dir, fname))
        if not match_info:
            continue
        if after:
            d = match_info['matchDate']
            # ISO dates compare correctly as text. A missing/odd date ('Unknown' sorts AFTER any digit!) must never pass.
            if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", d) or d < after:
                continue

        team1 = match_info['team1']
        winner = match_info['winner']

        q_data = generate_backtest_question(match_info)
        if not q_data:
            continue

        prob = q_data['probability']

        swapped_prob = None   # P(team2 wins) when team2 is named FIRST; stays NULL if that call fails (fill it later)
        if swap_teams:
            sw = generate_backtest_question({**match_info, 'team1': match_info['team2'], 'team2': match_info['team1']})
            swapped_prob = sw['probability'] if sw else None

        # Binary outcome mapping for team1 perspective
        if winner == team1:
            actual_outcome_str, actual_numeric = "YES", 1
        elif winner is None:
            actual_outcome_str, actual_numeric = "VOID", None
        else:
            actual_outcome_str, actual_numeric = "NO", 0

        if actual_numeric is not None:
            is_correct = 1 if ((actual_numeric == 1 and prob >= 0.5) or (actual_numeric == 0 and prob < 0.5)) else 0
            brier_score = (prob - actual_numeric) ** 2
        else:
            is_correct = None
            brier_score = None

        cursor.execute('''
            INSERT INTO backtest_predictions 
            (matchId, team1, team2, seriesName, match_format, match_date, venue,
             question, probability, 
             actual_outcome, is_correct, brier_score, created_at, model, prob_team2_first, prompt_variant)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            match_info['matchId'], team1, match_info['team2'], match_info['seriesName'],
            match_info['matchFormat'], match_info['matchDate'], match_info['venue'],
            q_data['question'], prob, actual_outcome_str, is_correct, brier_score,
            datetime.now(timezone.utc).isoformat(), MODEL, swapped_prob, _variant(),
        ))
        already_scored.add(match_info['matchId'])
        processed_count += 1
        print(f"[{processed_count}] {team1} vs {match_info['team2']} ({match_info['matchFormat']}) | Prob: {prob} | Actual Winner: {winner or 'Draw/No Result'} | Brier: {brier_score}" + (f" | swapped: {swapped_prob}" if swap_teams else ""))
        if processed_count % 10 == 0:
            conn.commit()   # keep progress if the run is interrupted

    conn.commit()
    conn.close()
    print(f"\n🎉 Simulation complete! Scored {processed_count} new matches into {db_path}.")
    return processed_count


# ---------------------------------------------------------------------------
# Position-bias backfill
# ---------------------------------------------------------------------------

def fill_swapped_probabilities(db_path: str, limit: int = 250) -> int:
    """For matches already scored by the CURRENT model, asks the swapped-order question (one call each) and stores
    prob_team2_first. Rows made by a different model (or by the notebook, model = NULL) are left alone: mixing
    models inside one comparison would defeat the purpose. Returns how many rows were filled."""
    _check_variant(db_path)
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        """SELECT id, team1, team2, seriesName, match_format, venue FROM backtest_predictions
           WHERE brier_score IS NOT NULL AND prob_team2_first IS NULL AND model = ? ORDER BY id LIMIT ?""",
        (MODEL, limit),
    ).fetchall()
    other = conn.execute(
        "SELECT COUNT(*) FROM backtest_predictions WHERE brier_score IS NOT NULL AND (model IS NULL OR model != ?)",
        (MODEL,),
    ).fetchone()[0]
    if other:
        print(f"Note: {other} scored row(s) were made by a different model (or the notebook) and will not be touched.")

    filled = 0
    for row_id, team1, team2, series, fmt, venue in rows:
        # same prompt, teams swapped: the answer is P(team2 wins) with team2 named first
        q = generate_backtest_question({"team1": team2, "team2": team1, "seriesName": series,
                                        "matchFormat": fmt, "venue": venue})
        if not q:
            continue
        conn.execute("UPDATE backtest_predictions SET prob_team2_first = ? WHERE id = ?", (q["probability"], row_id))
        filled += 1
        print(f"[{filled}/{len(rows)}] {team1} vs {team2} ({fmt}) | swapped-order P(team2 wins) = {q['probability']}")
        if filled % 10 == 0:
            conn.commit()
    conn.commit()
    conn.close()
    print(f"\nFilled the swapped-order probability for {filled} matches in {db_path}.")
    return filled


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def load_scored(db_path: str, model: str = None) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    query = "SELECT * FROM backtest_predictions WHERE brier_score IS NOT NULL"
    params = ()
    if model:
        query += " AND model = ?"
        params = (model,)
    df = pd.read_sql_query(query + " ORDER BY id ASC", conn, params=params)
    conn.close()
    if not df.empty:
        df['actual_numeric'] = df['actual_outcome'].apply(lambda x: 1 if x == 'YES' else 0)
    return df


def _bootstrap_ci(values, reps: int = 2000, seed: int = 0):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(reps, len(values)), replace=True).mean(axis=1)
    return np.percentile(means, [2.5, 97.5])


EDGES = (0.2, 0.4, 0.6, 0.8)
BIN_LABELS = ['0-20%', '20-40%', '40-60%', '60-80%', '80-100%']
MIN_BIN = 30     # a bin needs at least this many matches to count towards the "worst bin gap"


def _wilson(k: int, n: int, z: float = 1.96):
    """Wilson 95% interval for a win rate k/n (well behaved for small n, unlike the normal approximation)."""
    if n == 0:
        return float('nan'), float('nan')
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def _calibration_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per confidence bin: how many predictions, what the model said on average, what actually happened, the gap
    (actual minus predicted, in points) and a Wilson 95% interval on the actual rate. '*' marks a bin whose average
    prediction lies OUTSIDE that interval, i.e. a gap too large to be sampling noise."""
    d = df.copy()
    d['prob_bin'] = pd.cut(d['probability'], bins=[0.0] + list(EDGES) + [1.0], labels=BIN_LABELS)
    rows = []
    for label, g in d.groupby('prob_bin', observed=False):
        n = len(g)
        if n == 0:
            rows.append({'prob_bin': label, 'total_predictions': 0, 'mean_predicted_prob': np.nan, 'actual_win_rate': np.nan,
                         'gap_pts': np.nan, 'actual_95ci': '--', 'real_gap': ''})
            continue
        k = int((g['actual_outcome'] == 'YES').sum())
        pred, act = g['probability'].mean(), k / n
        lo, hi = _wilson(k, n)
        rows.append({'prob_bin': label, 'total_predictions': n, 'mean_predicted_prob': round(pred * 100, 1),
                     'actual_win_rate': round(act * 100, 1), 'gap_pts': round((act - pred) * 100, 1),
                     'actual_95ci': f"[{lo * 100:.1f}, {hi * 100:.1f}]", 'real_gap': '*' if not (lo <= pred <= hi) else ''})
    return pd.DataFrame(rows)


def plot_reliability(df: pd.DataFrame, path: str = "reliability.png"):
    """Reliability diagram: each bin's actual win rate (with its Wilson 95% interval) against what the model said.
    A bin whose interval does not touch the diagonal has a gap that is not sampling noise."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(reliability plot skipped: matplotlib not installed)")
        return
    t = _calibration_table(df)
    t = t[t['total_predictions'] > 0]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 100], [0, 100], 'k--', lw=1, label='perfect calibration')
    x, yv = t['mean_predicted_prob'].to_numpy(float), t['actual_win_rate'].to_numpy(float)
    lo = np.array([float(s.strip('[]').split(',')[0]) for s in t['actual_95ci']])
    hi = np.array([float(s.strip('[]').split(',')[1]) for s in t['actual_95ci']])
    ax.errorbar(x, yv, yerr=[yv - lo, hi - yv], fmt='o-', color='tab:orange', capsize=4, label='actual rate, Wilson 95% CI')
    for xi, yi, ni in zip(x, yv, t['total_predictions']):
        ax.annotate(f"n={ni}", (xi, yi), textcoords='offset points', xytext=(6, -14), fontsize=8)
    ax.set_xlim(0, 100); ax.set_ylim(0, 100)
    ax.set_xlabel('Mean predicted probability (%)'); ax.set_ylabel('Actual win rate (%)')
    ax.set_title(f'Reliability ({len(df)} matches)'); ax.grid(alpha=.3); ax.legend(loc='upper left')
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    print(f"Saved reliability plot to {path}")


def evaluate_calibration(df: pd.DataFrame):
    if df.empty:
        print("No scored backtest data found in database yet.")
        return

    mean_brier = df['brier_score'].mean()
    accuracy = df['is_correct'].mean() * 100
    brier_lo, brier_hi = _bootstrap_ci(df['brier_score'])
    acc_lo, acc_hi = _bootstrap_ci(df['is_correct'] * 100)
    base_rate = df['actual_numeric'].mean()
    base_rate_brier = base_rate * (1 - base_rate)

    models = df['model'].dropna().unique().tolist() if 'model' in df else []
    print("==========================================")
    print("      BACKTEST PERFORMANCE METRICS        ")
    print("==========================================")
    print(f"Model(s)                : {models or 'not recorded'}")
    print(f"Total Evaluated Matches : {len(df)}")
    print(f"Mean Brier Score        : {mean_brier:.4f}   95% CI [{brier_lo:.4f}, {brier_hi:.4f}]")
    print(f"  baselines             : constant 50% = 0.2500 | always the team1 base rate ({base_rate:.1%}) = {base_rate_brier:.4f}")
    print(f"Directional Accuracy    : {accuracy:.2f}%   95% CI [{acc_lo:.1f}%, {acc_hi:.1f}%]")
    print("==========================================\n")

    # Calibration breakdown across confidence intervals
    print("Calibration Table (Model Confidence vs Actual Win Rate):")
    print(_calibration_table(df).to_string(index=False))

    print("\nBy match format:")
    by_format = df.groupby('match_format').agg(
        matches=('brier_score', 'count'),
        mean_brier=('brier_score', 'mean'),
        accuracy_pct=('is_correct', lambda x: x.mean() * 100),
    ).reset_index().sort_values('matches', ascending=False)
    print(by_format.round(4).to_string(index=False))
    print()


def evaluate_position_bias(df: pd.DataFrame):
    """Compares team1's probability when named FIRST (the original prompt) with when named SECOND (swapped prompt).
    If the model were order-neutral the two would agree; the gap is the 'first-named advantage'."""
    if 'prob_team2_first' not in df.columns:
        return
    d = df.dropna(subset=['prob_team2_first']).copy()
    if len(d) < 20:
        print("Position-bias check: fewer than 20 matches scored in both team orders. "
              "Run with --fill-swaps (matches already in the DB) or --swap-teams (new matches).\n")
        return

    n = len(d)
    y = d['actual_numeric'].astype(float)
    p_first = d['probability'].astype(float)                    # P(team1 wins), team1 named first
    p_second = 1.0 - d['prob_team2_first'].astype(float)        # P(team1 wins), team1 named second
    p_avg = (p_first + p_second) / 2.0                          # order-averaged estimate
    adv = p_first - p_second                                    # extra probability a team gets for being named first

    mean_adv = adv.mean()
    se = adv.std(ddof=1) / np.sqrt(n)
    z = mean_adv / se if se > 0 else 0.0
    p_value = math.erfc(abs(z) / math.sqrt(2))

    b_first, b_second, b_avg = ((p_first - y) ** 2), ((p_second - y) ** 2), ((p_avg - y) ** 2)
    gain = b_first - b_avg
    g_lo, g_hi = _bootstrap_ci(gain.values)
    acc = lambda p: ((p >= 0.5).astype(float) == y).mean() * 100
    flips = ((p_first >= 0.5) != (p_second >= 0.5)).mean() * 100

    print("==========================================")
    print("   POSITION-BIAS CHECK (order of teams)   ")
    print("==========================================")
    print(f"Matches scored in both orders : {n}")
    print(f"Mean P(team1 wins)            : named first {p_first.mean():.1%} | named second {p_second.mean():.1%} "
          f"| order-averaged {p_avg.mean():.1%} | actual {y.mean():.1%}")
    print(f"First-named advantage         : {mean_adv * 100:+.1f} pts   95% CI [{(mean_adv - 1.96 * se) * 100:+.1f}, {(mean_adv + 1.96 * se) * 100:+.1f}]"
          f"   z = {z:.1f}, p = {p_value:.3g}")
    print("   (0 = order-neutral; positive = the model favours whichever team it is told about first)")
    print(f"Brier                         : named first {b_first.mean():.4f} | named second {b_second.mean():.4f} | order-averaged {b_avg.mean():.4f}")
    print(f"  gain from order-averaging   : {gain.mean():+.4f}   95% CI [{g_lo:+.4f}, {g_hi:+.4f}]   (positive = averaging helps)")
    print(f"Accuracy                      : named first {acc(p_first):.1f}% | named second {acc(p_second):.1f}% | order-averaged {acc(p_avg):.1f}%")
    print(f"Swapping the order changes the predicted favourite in {flips:.1f}% of matches")
    print("==========================================\n")

    d2 = d.copy()
    d2['probability'] = p_avg
    print("Calibration of the ORDER-AVERAGED probabilities:")
    print(_calibration_table(d2).to_string(index=False))
    print()


def _fit_platt(train_df: pd.DataFrame):
    from sklearn.linear_model import LogisticRegression
    eps = 1e-5
    probs = np.clip(train_df['probability'], eps, 1 - eps)
    logits = np.log(probs / (1 - probs)).values.reshape(-1, 1)
    calibrator = LogisticRegression()
    calibrator.fit(logits, train_df['actual_numeric'])
    return calibrator


def _apply_platt(calibrator, df: pd.DataFrame):
    eps = 1e-5
    probs = np.clip(df['probability'], eps, 1 - eps)
    logits = np.log(probs / (1 - probs)).values.reshape(-1, 1)
    return calibrator.predict_proba(logits)[:, 1]


def evaluate_with_platt_scaling(df: pd.DataFrame):
    if len(df) < 20:
        print("Need at least 20 evaluated matches to fit a calibration model.")
        return

    # Train on the first 60% (insertion order), test on the remaining 40%
    train_size = int(len(df) * 0.6)
    train_df = df.iloc[:train_size]
    test_df = df.iloc[train_size:].copy()

    calibrator = _fit_platt(train_df)
    test_df['calibrated_prob'] = _apply_platt(calibrator, test_df)
    test_df['raw_brier'] = (test_df['probability'] - test_df['actual_numeric']) ** 2
    test_df['calibrated_brier'] = (test_df['calibrated_prob'] - test_df['actual_numeric']) ** 2

    print("==========================================")
    print("      PLATT SCALING CALIBRATION TEST      ")
    print("==========================================")
    print(f"Training Samples : {len(train_df)}")
    print(f"Test Samples     : {len(test_df)}")
    print(f"Raw Brier Score  : {test_df['raw_brier'].mean():.4f}")
    print(f"Calibrated Brier : {test_df['calibrated_brier'].mean():.4f} (Lower is better!)")
    r_ece = _ece(test_df['probability'].to_numpy(float), test_df['actual_numeric'].to_numpy(float))
    c_ece = _ece(test_df['calibrated_prob'].to_numpy(float), test_df['actual_numeric'].to_numpy(float))
    print(f"ECE on this test set: raw {r_ece * 100:.1f} pts -> calibrated {c_ece * 100:.1f} pts "
          f"(one split = one draw; the out-of-fold CALIBRATION METHODS section below is the evidence)")
    print("==========================================\n")


def _ece(p, y, edges=(0.2, 0.4, 0.6, 0.8)) -> float:
    """Expected calibration error with the same 5 bins as the calibration table: sum_k share_k * |mean pred - mean actual|."""
    idx = np.digitize(p, edges, right=True)      # (0,0.2], (0.2,0.4], ... like pd.cut
    total = 0.0
    for k in range(len(edges) + 1):
        m = idx == k
        if m.any():
            total += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(total)


def _fit_calibrator(method: str, p_train, y_train):
    eps = 1e-5
    if method == "platt":
        from sklearn.linear_model import LogisticRegression
        pc = np.clip(p_train, eps, 1 - eps)
        return LogisticRegression().fit(np.log(pc / (1 - pc)).reshape(-1, 1), y_train)
    if method == "isotonic":
        from sklearn.isotonic import IsotonicRegression
        # y_min/y_max keep outputs in the same 0.01-0.99 range the model itself uses (isotonic can otherwise emit exact 0 or 1)
        return IsotonicRegression(y_min=0.01, y_max=0.99, out_of_bounds="clip").fit(p_train, y_train)
    raise ValueError(method)


def _apply_calibrator(method: str, model, p):
    if method == "platt":
        pc = np.clip(p, 1e-5, 1 - 1e-5)
        return model.predict_proba(np.log(pc / (1 - pc)).reshape(-1, 1))[:, 1]
    return model.predict(p)


def _bin_stats(p, y, edges=EDGES):
    """Per confidence bin: (n, mean prediction, actual rate). Same bins as _ece and the calibration table."""
    idx = np.digitize(p, edges, right=True)
    out = []
    for k in range(len(edges) + 1):
        m = idx == k
        n = int(m.sum())
        out.append((n, float(p[m].mean()), float(y[m].mean())) if n else (0, float('nan'), float('nan')))
    return out


def _worst_gap(p, y, min_n: int = MIN_BIN) -> float:
    """Largest |actual - predicted| over bins holding at least min_n matches: the number a user who picks one bin cares about."""
    gaps = [abs(pm - ym) for n, pm, ym in _bin_stats(p, y) if n >= min_n]
    return max(gaps) if gaps else float('nan')


def _ece_null_draws(p, sims: int = 1000, n_sub: int = None, seed: int = 0):
    """ECEs of `sims` simulated worlds in which the probabilities are PERFECTLY calibrated: outcomes are drawn as Bernoulli(p).
    Whatever ECE these show is sampling noise, nothing else."""
    rng = np.random.default_rng(seed)
    p = np.asarray(p, float)
    vals = []
    for _ in range(sims):
        pp = p if (n_sub is None or n_sub >= len(p)) else rng.choice(p, size=n_sub, replace=False)
        vals.append(_ece(pp, (rng.random(len(pp)) < pp).astype(float)))
    return np.array(vals)


def _ece_noise_floor(p, sims: int = 1000, n_sub: int = None, seed: int = 0):
    """The ECE you get from sampling noise alone. Returns (mean, 95th percentile). An ECE at or below this is
    indistinguishable from perfect calibration."""
    vals = _ece_null_draws(p, sims=sims, n_sub=n_sub, seed=seed)
    return float(np.mean(vals)), float(np.percentile(vals, 95))


def _pipeline_null(p, y, methods, worlds: int = 100, folds: int = 5, seed: int = 0):
    """An apples-to-apples yardstick for the out-of-fold ECE. Out-of-fold ECE is OPTIMISTIC: the calibrator is fitted on the
    same matches whose outcomes it is scored against, so it partly absorbs this sample's luck, and it lands BELOW the plain
    noise floor even when nothing is wrong. So: suppose the Platt-calibrated probabilities were exactly right, simulate new
    outcomes from them, and run the SAME pipeline. Returns, per method, the out-of-fold ECE it would show (mean, 95th pct)
    and the ECE a calibrator fitted here would show on NEW matches of the same size (the number to expect in the field),
    plus the ECE the raw probabilities would show on new matches in that world."""
    from sklearn.model_selection import StratifiedKFold
    rng = np.random.default_rng(seed + 11)
    n = len(p)
    q = _apply_calibrator("platt", _fit_calibrator("platt", p, y), p)      # the 'true' probability in the simulated world
    oof_e = {m: [] for m in methods}
    new_e = {m: [] for m in methods}
    raw_new = []
    for _ in range(worlds):
        ys = (rng.random(n) < q).astype(float)                              # outcomes the calibrator gets to see
        yn = (rng.random(n) < q).astype(float)                              # fresh outcomes for the same kind of matches
        if len(np.unique(ys)) < 2:
            continue
        o = {m: np.empty(n) for m in methods}
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=int(rng.integers(2**31 - 1)))
        for tr, te in skf.split(p, ys):
            for m in methods:
                o[m][te] = _apply_calibrator(m, _fit_calibrator(m, p[tr], ys[tr]), p[te])
        for m in methods:
            oof_e[m].append(_ece(o[m], ys))
            new_e[m].append(_ece(_apply_calibrator(m, _fit_calibrator(m, p, ys), p), yn))
        raw_new.append(_ece(p, yn))
    out = {m: {"oof_mean": float(np.mean(oof_e[m])), "oof_95": float(np.percentile(oof_e[m], 95)),
               "new_mean": float(np.mean(new_e[m])), "new_95": float(np.percentile(new_e[m], 95))} for m in methods}
    return out, float(np.mean(raw_new))


def run_calibration_experiment(df: pd.DataFrame, repeats: int = 40, folds: int = 5, seed: int = 0, boot: int = 500):
    """Raw vs Platt vs isotonic, judged on CALIBRATION (ECE and the per-bin gaps), scored OUT-OF-FOLD with repeated
    stratified k-fold: every match is predicted by a calibrator that never saw it, so no single lucky split decides the
    answer. Brier is reported last, as a check that calibrating did no harm (it can only move by the small REL term)."""
    from sklearn.model_selection import StratifiedKFold
    p = df['probability'].to_numpy(float)
    y = df['actual_numeric'].to_numpy(float)
    n = len(p)
    if n < 100:
        print(f"Calibration experiment needs at least 100 scored matches (have {n}).")
        return None
    methods = ("platt", "isotonic")
    rng = np.random.default_rng(seed)
    raw_sq = (p - y) ** 2
    raw_brier, raw_ece = raw_sq.mean(), _ece(p, y)

    oof = {m: np.empty((repeats, n)) for m in methods}          # out-of-fold prediction of every match, in every repeat
    for r in range(repeats):
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=int(rng.integers(2**31 - 1)))
        for tr, te in skf.split(p, y):
            for m in methods:
                oof[m][r, te] = _apply_calibrator(m, _fit_calibrator(m, p[tr], y[tr]), p[te])

    ece_runs = {m: np.array([_ece(oof[m][r], y) for r in range(repeats)]) for m in methods}
    worst = {m: float(np.nanmean([_worst_gap(oof[m][r], y) for r in range(repeats)])) for m in methods}
    raw_worst = _worst_gap(p, y)
    null = _ece_null_draws(p)                                   # ECE of 1000 perfectly calibrated simulated worlds
    floor_mean, floor_95 = float(null.mean()), float(np.percentile(null, 95))
    p_raw = (1 + int((null >= raw_ece).sum())) / (len(null) + 1)   # P(noise alone gives an ECE this large)

    # POINT reductions: plain subtraction of the out-of-fold ECEs. These are the headline numbers.
    point = {m: raw_ece - float(ece_runs[m].mean()) for m in methods}
    point_diff = float(ece_runs["platt"].mean() - ece_runs["isotonic"].mean())    # positive = isotonic better

    # Paired bootstrap over matches: the SAME resampled matches are scored raw and calibrated. (Uses ~10 repeats for speed.)
    # CAUTION: resampling with replacement duplicates matches and adds noise, which inflates the calibrated ECE (already at
    # the noise floor) more than the raw ECE, so the bootstrap MEAN reduction comes out too small. The interval below is
    # therefore re-centred: shifted by (point estimate - bootstrap mean). It is approximate, not exact.
    brng = np.random.default_rng(seed + 7)
    use = list(range(0, repeats, max(1, repeats // 10)))
    red = {m: [] for m in methods}
    iso_vs_platt = []
    for _ in range(boot):
        idx = brng.integers(0, n, n)
        yb = y[idx]
        rb = _ece(p[idx], yb)
        e = {m: float(np.mean([_ece(oof[m][r][idx], yb) for r in use])) for m in methods}
        for m in methods:
            red[m].append(rb - e[m])
        iso_vs_platt.append(e["platt"] - e["isotonic"])

    print("==================================================")
    print(" CALIBRATION METHODS: raw vs Platt vs isotonic")
    print(f" {n} matches; {repeats} repeats of {folds}-fold cross-validation; every calibrated figure is out-of-fold")
    print("==================================================")
    print(f"ECE from sampling noise alone with these probabilities: {floor_mean * 100:.1f} pts (95th percentile {floor_95 * 100:.1f})")
    print("  -> for probabilities used AS THEY ARE (not fitted to this sample), an ECE at or below this cannot be told apart from perfect calibration")
    p_txt = f"< {1 / (len(null) + 1):.3f}" if (null >= raw_ece).sum() == 0 else f"= {p_raw:.3f}"
    print(f"Is the RAW miscalibration real? Raw ECE {raw_ece * 100:.1f} pts; share of perfectly calibrated simulated worlds "
          f"with an ECE that large: p {p_txt}  ({'real' if p_raw < 0.05 else 'not distinguishable from noise'})\n")
    print(f"{'method':10s} {'ECE':>10s} {'reduction':>14s}   {'95% CI (re-centred)':22s} {'worst bin gap':>13s}")
    print(f"{'raw':10s} {raw_ece * 100:6.1f} pts {'--':>14s}   {'--':22s} {raw_worst * 100:9.1f} pts")
    for m in methods:
        lo, hi = np.percentile(red[m], [2.5, 97.5])
        shift = point[m] - float(np.mean(red[m]))
        ci = f"[{(lo + shift) * 100:+.1f}, {(hi + shift) * 100:+.1f}]"
        print(f"{m:10s} {ece_runs[m].mean() * 100:6.1f} pts {point[m] * 100:+10.1f} pts   {ci:22s} {worst[m] * 100:9.1f} pts")
    print("  reduction = raw ECE minus out-of-fold ECE (plain subtraction). The bootstrap on its own understates it "
          f"(its mean reduction was {', '.join(f'{m} {np.mean(red[m]) * 100:+.1f}' for m in methods)}), because resampling adds noise that inflates the "
          "calibrated ECE more than the raw one; the interval is shifted by that bias, so treat it as approximate.")
    d_lo, d_hi = np.percentile(iso_vs_platt, [2.5, 97.5])
    d_shift = point_diff - float(np.mean(iso_vs_platt))
    d_lo, d_hi = d_lo + d_shift, d_hi + d_shift
    print(f"\nPlatt ECE minus isotonic ECE: {point_diff * 100:+.1f} pts   95% CI [{d_lo * 100:+.1f}, {d_hi * 100:+.1f}]"
          f"   (positive = isotonic better calibrated; an interval containing 0 = no difference)")
    print(f"(worst bin gap = largest |actual - predicted| among bins with >= {MIN_BIN} matches)")

    # Like-for-like yardstick: out-of-fold ECE is flattered (the calibrator is fitted to this sample), so it must NOT be
    # compared with the plain noise floor above. Compare it with what this same pipeline gives if calibration were perfect.
    nul, raw_new = _pipeline_null(p, y, methods, seed=seed)
    print("\nIf the Platt-calibrated probabilities were exactly right, the SAME pipeline would show (mean, 95th percentile of 100 simulated worlds):")
    print(f"{'method':10s} {'out-of-fold ECE':>22s}   {'observed':>9s}   {'ECE on NEW matches':>22s}")
    for m in methods:
        print(f"{m:10s} {nul[m]['oof_mean'] * 100:8.1f} / {nul[m]['oof_95'] * 100:4.1f} pts   {ece_runs[m].mean() * 100:6.1f} pts   "
              f"{nul[m]['new_mean'] * 100:8.1f} / {nul[m]['new_95'] * 100:4.1f} pts")
    print(f"  Compare the observed out-of-fold ECE with the 'out-of-fold' column: at or below it = no sign of leftover miscalibration.")
    print(f"  The 'NEW matches' column is the ECE to EXPECT on fresh matches of this size (raw probabilities would show about "
          f"{raw_new * 100:.1f} there); out-of-fold ECE is optimistic and is not what a new sample will show.")

    # Per-bin gaps, averaged over the repeats: the 'did the 20-40% bin get fixed' table
    raw_bins = _bin_stats(p, y)
    print("\nPer-bin gap, actual minus predicted in points (n = matches in the bin; calibrated bins use the calibrated probability):")
    print(f"{'bin':8s} | {'raw n':>5s} {'gap':>6s} | {'Platt n':>7s} {'gap':>6s} | {'isotonic n':>10s} {'gap':>6s}")
    stats = {m: [_bin_stats(oof[m][r], y) for r in range(repeats)] for m in methods}
    for k, label in enumerate(BIN_LABELS):
        rn, rp, ry = raw_bins[k]
        cells = []
        for m in methods:
            ns = [s[k][0] for s in stats[m]]
            gs = [s[k][2] - s[k][1] for s in stats[m] if s[k][0] > 0]
            cells.append((np.mean(ns), np.mean(gs) * 100 if gs else float('nan')))
        rg = (ry - rp) * 100 if rn else float('nan')
        print(f"{label:8s} | {rn:5d} {rg:+6.1f} | {cells[0][0]:7.0f} {cells[0][1]:+6.1f} | {cells[1][0]:10.0f} {cells[1][1]:+6.1f}")

    # Platt's own parameters: slope < 1 means the model is over-confident (its log-odds must be shrunk)
    plat = _fit_calibrator("platt", p, y)
    print(f"\nPlatt fit on all {n} matches: calibrated log-odds = {plat.intercept_[0]:+.2f} + {plat.coef_[0][0]:.2f} x model log-odds"
          f"   (slope < 1 = over-confident, > 1 = under-confident; fitted in-sample, so descriptive only)")

    # Brier last: a no-harm check, not the target
    print("\nBrier (no-harm check; calibration can only move the REL part, a few thousandths):")
    print(f"{'raw':10s} {raw_brier:.4f}")
    for m in methods:
        sq = ((oof[m] - y) ** 2).mean(axis=0)                 # per-match squared error, averaged over repeats
        gain = raw_sq - sq
        lo, hi = _bootstrap_ci(gain)
        print(f"{m:10s} {sq.mean():.4f}   gain {gain.mean():+.4f}   95% CI [{lo:+.4f}, {hi:+.4f}]")
    print("==================================================\n")
    return {"raw_ece": raw_ece, "ece": {m: float(ece_runs[m].mean()) for m in methods},
            "ece_reduction": point, "ece_reduction_bootstrap_mean": {m: float(np.mean(red[m])) for m in methods},
            "platt_minus_isotonic": point_diff, "noise_floor": floor_mean, "raw_ece_p_value": p_raw,
            "pipeline_null": nul, "raw_ece_new_matches": raw_new}


def compare_prompt_variants(df: pd.DataFrame, old_db: str, model: str = None):
    """Paired comparison of THIS db's predictions with another db's for the SAME matches (no API calls). Typical use: the
    old run that put the series name in the prompt against the re-run without it, to measure how much that name helped."""
    old = load_scored(old_db, model=model)
    if old.empty or df.empty:
        print("compare: one of the databases has no scored rows.")
        return
    m = df[['matchId', 'probability', 'actual_numeric']].merge(
        old[['matchId', 'probability']], on='matchId', suffixes=('_new', '_old'))
    if len(m) < 30:
        print(f"compare: only {len(m)} matches in common with {old_db}; need at least 30.")
        return
    y = m['actual_numeric'].values.astype(float)
    pn, po = m['probability_new'].values, m['probability_old'].values
    print("\n" + "=" * 62)
    print(f"  PAIRED COMPARISON on {len(m)} matches scored in both databases")
    print(f"  A = {old_db}    B = this database")
    print("=" * 62)
    for name, p in (("A", po), ("B", pn)):
        print(f"{name}: Brier {np.mean((p - y) ** 2):.4f} | accuracy {np.mean((p >= 0.5) == (y == 1)):.1%} | "
              f"ECE {_ece(p, y) * 100:.1f} pts | worst bin gap {_worst_gap(p, y) * 100:.1f} pts")
    d = (pn - y) ** 2 - (po - y) ** 2
    lo, hi = _bootstrap_ci(d)
    print(f"Brier B - A: {d.mean():+.4f}   95% CI [{lo:+.4f}, {hi:+.4f}]   (positive = B is worse)")
    da = ((pn >= 0.5) == (y == 1)).astype(float) - ((po >= 0.5) == (y == 1)).astype(float)
    lo, hi = _bootstrap_ci(da)
    print(f"accuracy B - A: {da.mean() * 100:+.1f} pts   95% CI [{lo * 100:+.1f}, {hi * 100:+.1f}]")
    print(f"mean |probability change| per match: {np.mean(np.abs(pn - po)):.3f}; "
          f"the predicted favourite changes in {np.mean((pn >= 0.5) != (po >= 0.5)):.1%} of matches")
    print("=" * 62)


def run_learning_curve(df: pd.DataFrame, sizes=(100, 200, 300, 400, 500, 700, 1000, 1500), repeats: int = 20,
                       folds: int = 5, seed: int = 0):
    """Does more calibration data help? Same out-of-fold protocol as the main experiment (every match is scored by a calibrator
    that never saw it, so ECE is measured on ALL the matches and its noise floor is the same at every size), except that
    the calibrator is fitted on a random subset of only `size` matches of each training fold. A calibrated ECE that keeps
    falling as size grows is a method still starved for data; one that has reached the noise floor is not."""
    from sklearn.model_selection import StratifiedKFold
    p = df['probability'].to_numpy(float)
    y = df['actual_numeric'].to_numpy(float)
    n = len(p)
    usable = [s for s in sizes if s <= int(n * (folds - 1) / folds) - 2]
    if not usable:
        print("Learning curve skipped: not enough matches.")
        return None
    methods = ("platt", "isotonic")
    rng = np.random.default_rng(seed + 1)
    raw_ece, raw_brier = _ece(p, y), ((p - y) ** 2).mean()
    floor_mean, floor_95 = _ece_noise_floor(p, seed=seed + 2)
    print("==================================================")
    print(" LEARNING CURVE: out-of-fold ECE vs calibration training size")
    print(f" ({repeats} repeats of {folds}-fold CV; the calibrator sees only `train size` matches; all {n} matches are scored)")
    print(f" raw ECE {raw_ece * 100:.1f} pts | ECE from sampling noise alone {floor_mean * 100:.1f} pts (95th percentile {floor_95 * 100:.1f})")
    print("==================================================")
    print(f"{'train size':>10s} | {'Platt ECE':>10s} {'isotonic ECE':>13s} | {'Platt Brier gain':>16s} {'iso Brier gain':>15s}")
    rows = []
    for size in usable:
        ece = {m: [] for m in methods}
        bgain = {m: [] for m in methods}
        for _ in range(repeats):
            skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=int(rng.integers(2**31 - 1)))
            oof = {m: np.empty(n) for m in methods}
            for tr, te in skf.split(p, y):
                sub = rng.choice(tr, size=size, replace=False)
                if len(np.unique(y[sub])) < 2:
                    sub = tr
                for m in methods:
                    oof[m][te] = _apply_calibrator(m, _fit_calibrator(m, p[sub], y[sub]), p[te])
            for m in methods:
                ece[m].append(_ece(oof[m], y))
                bgain[m].append(raw_brier - ((oof[m] - y) ** 2).mean())
        print(f"{size:10d} | {np.mean(ece['platt']) * 100:8.1f}pt {np.mean(ece['isotonic']) * 100:11.1f}pt | "
              f"{np.mean(bgain['platt']):+16.4f} {np.mean(bgain['isotonic']):+15.4f}")
        rows.append({"size": size, "raw_ece": float(raw_ece), "platt_ece": float(np.mean(ece['platt'])),
                     "isotonic_ece": float(np.mean(ece['isotonic'])), "noise_floor": floor_mean,
                     "platt_brier_gain": float(np.mean(bgain['platt'])), "isotonic_brier_gain": float(np.mean(bgain['isotonic']))})
    print("==================================================\n")
    return rows


def run_expanding_calibration(df: pd.DataFrame, chunk_size: int = 40, plot_path: str = "expanding_window.png"):
    """Walk through the matches in order. Each chunk is scored by calibrators fitted only on the chunks before it. The
    report is CUMULATIVE ECE: every out-of-sample prediction made so far, pooled, raw vs calibrated. (ECE of a single
    40-match chunk is pure noise, so chunks are only used to build the pool.)"""
    N = len(df)
    if N < chunk_size * 3:
        print(f"Found {N} matches. You need at least {chunk_size * 3} for the expanding-window test (3+ chunks).")
        return None
    p = df['probability'].to_numpy(float)
    y = df['actual_numeric'].to_numpy(float)
    methods = ("platt", "isotonic")
    num_chunks = N // chunk_size
    oos = {m: p.copy() for m in methods}
    records = []
    print("==================================================")
    print(f" EXPANDING WINDOW: cumulative out-of-sample ECE ({N} matches, chunks of {chunk_size})")
    print("==================================================")
    print(f"{'step':>4s} {'train':>6s} {'pooled':>7s} | {'raw ECE':>8s} {'Platt':>7s} {'isotonic':>9s} | {'noise floor':>11s}")
    for i in range(1, num_chunks):
        s, e = i * chunk_size, (i + 1) * chunk_size
        if len(np.unique(y[:s])) >= 2:
            for m in methods:
                oos[m][s:e] = _apply_calibrator(m, _fit_calibrator(m, p[:s], y[:s]), p[s:e])
        sel = slice(chunk_size, e)            # chunk 0 is the warm-up (no calibrator yet), so it is left out of the pool
        pooled = e - chunk_size
        r_ece, pl_ece, is_ece = _ece(p[sel], y[sel]), _ece(oos["platt"][sel], y[sel]), _ece(oos["isotonic"][sel], y[sel])
        fl, _ = _ece_noise_floor(p[sel], sims=300, seed=i)
        print(f"{i + 1:4d} {s:6d} {pooled:7d} | {r_ece * 100:6.1f}pt {pl_ece * 100:5.1f}pt {is_ece * 100:7.1f}pt | {fl * 100:9.1f}pt")
        records.append({"Step": i + 1, "Training Size": s, "Pooled": pooled, "Raw ECE": r_ece, "Platt ECE": pl_ece,
                        "Isotonic ECE": is_ece, "Noise floor": fl})
    results = pd.DataFrame(records)

    sel = slice(chunk_size, num_chunks * chunk_size)
    raw_b = (p[sel] - y[sel]) ** 2
    print("--------------------------------------------------")
    print(f"Pooled over all {num_chunks * chunk_size - chunk_size} out-of-sample matches: raw Brier {raw_b.mean():.4f}, "
          f"Platt {((oos['platt'][sel] - y[sel]) ** 2).mean():.4f}, isotonic {((oos['isotonic'][sel] - y[sel]) ** 2).mean():.4f} "
          f"(no-harm check only).")
    print("==================================================\n")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(results['Training Size'], results['Raw ECE'] * 100, 'o--', color='tab:red', label='raw')
        ax.plot(results['Training Size'], results['Platt ECE'] * 100, 's-', color='tab:green', label='Platt (out-of-sample)')
        ax.plot(results['Training Size'], results['Isotonic ECE'] * 100, '^-', color='tab:blue', label='isotonic (out-of-sample)')
        ax.plot(results['Training Size'], results['Noise floor'] * 100, color='gray', ls=':', label='ECE from sampling noise alone')
        ax.set_xlabel('Matches the calibrator was trained on'); ax.set_ylabel('Cumulative ECE over all out-of-sample matches so far (points)')
        ax.set_title('Expanding window: does calibration error fall as the calibrator sees more data?')
        ax.grid(alpha=.3); ax.legend(); fig.tight_layout(); fig.savefig(plot_path, dpi=120); plt.close(fig)
        print(f"Saved plot to {plot_path}")
    except Exception as e:
        print(f"(Plot skipped: {e})")
    return results


def main():
    ap = argparse.ArgumentParser(description="Cricsheet backtest (DeepSeek).")
    ap.add_argument("--data-dir", default=DEFAULT_DATA_DIR, help="folder of Cricsheet JSON files")
    ap.add_argument("--db", default=DEFAULT_DB)
    ap.add_argument("--n", type=int, default=250, help="how many NEW matches to score this run (0 = just report)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--eval-only", action="store_true", help="skip scoring; report from the DB (no API calls)")
    ap.add_argument("--model", default=None, help="only evaluate rows made by this model (default: all rows)")
    ap.add_argument("--after", default=None, help="only score matches dated on/after YYYY-MM-DD (set to the model's training cutoff + margin; use a separate --db)")
    ap.add_argument("--chunk-size", type=int, default=40)
    ap.add_argument("--calibration-experiment", action="store_true", help="raw vs Platt vs isotonic, out-of-fold, plus a learning curve (no API calls)")
    ap.add_argument("--calibration-only", action="store_true", help="print ONLY the calibration experiment and learning curve (skips the long report, the single-split test and the expanding window); implies --eval-only and --calibration-experiment, so no API calls")
    ap.add_argument("--repeats", type=int, default=40, help="repeats of 5-fold cross-validation in the calibration experiment")
    ap.add_argument("--boot", type=int, default=500, help="bootstrap resamples for the interval on the ECE reduction")
    ap.add_argument("--reliability-plot", default="reliability.png", help="where to save the reliability diagram with Wilson intervals ('' = skip)")
    ap.add_argument("--prob-source", choices=["raw", "avg"], default="raw", help="probability to calibrate: stored P(team1) or the order-averaged one (needs swapped probabilities)")
    ap.add_argument("--swap-teams", action="store_true", help="score NEW matches in both team orders (2 calls per match) for the position-bias check")
    ap.add_argument("--fill-swaps", action="store_true", help="add the swapped-order probability to matches ALREADY in the DB (1 call each, current model only); --n caps how many")
    ap.add_argument("--no-series-name", action="store_true", help="leave the Cricsheet series name out of the prompt (ablation; use a separate --db)")
    ap.add_argument("--match-ids-from", default=None, help="score exactly the matches stored in this other DB (use with a NEW --db and a large --n) instead of a random draw")
    ap.add_argument("--compare-with", default=None, help="after reporting, compare with this other DB on the matches both scored (no API calls; add --model if it holds several)")
    args = ap.parse_args()
    global INCLUDE_SERIES
    INCLUDE_SERIES = not args.no_series_name
    if args.calibration_only:                 # a pure reporting mode: never calls the API, always runs the experiment
        args.eval_only = True
        args.calibration_experiment = True

    init_backtest_db(args.db)
    if args.fill_swaps:
        fill_swapped_probabilities(args.db, limit=args.n if args.n > 0 else 250)
    elif not args.eval_only and args.n > 0:
        run_backtest_simulation(args.data_dir, args.db, n=args.n, seed=args.seed, after=args.after, swap_teams=args.swap_teams,
                                ids_from=args.match_ids_from)

    df = load_scored(args.db, model=args.model)
    if args.compare_with:
        compare_prompt_variants(df, args.compare_with, model=args.model)
    if not args.calibration_only:
        evaluate_calibration(df)
    if not df.empty:
        if args.reliability_plot:
            plot_reliability(df, args.reliability_plot)
        if not args.calibration_only:
            evaluate_position_bias(df)
    try:
        if not args.calibration_only:
            evaluate_with_platt_scaling(df)
            run_expanding_calibration(df, chunk_size=args.chunk_size)
        if args.calibration_experiment and not df.empty:
            d = df.copy()
            if args.prob_source == "avg":
                have = d.dropna(subset=['prob_team2_first']) if 'prob_team2_first' in d.columns else d.iloc[0:0]
                if len(have) < 100:
                    print("--prob-source avg needs at least 100 matches with swapped probabilities (run --fill-swaps first); using raw.")
                else:
                    d = have.copy()
                    d['probability'] = (d['probability'] + 1.0 - d['prob_team2_first']) / 2.0
                    print(f"Calibration experiment on the ORDER-AVERAGED probabilities ({len(d)} matches).")
            run_calibration_experiment(d, repeats=args.repeats, seed=args.seed, boot=args.boot)
            run_learning_curve(d, seed=args.seed)
    except ImportError:
        print("scikit-learn is not installed: skipping Platt scaling / expanding window / calibration experiment (pip install scikit-learn).")


if __name__ == "__main__":
    main()