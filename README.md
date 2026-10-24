# Part B — Cricket Prediction Engine

An end-to-end prediction pipeline for cricket: pull real fixtures, generate a
yes/no question with a probability and a resolution rule, lock the market shortly
before the match starts, auto-resolve from the final result, and score itself
with a Brier score.

Cricket is the grounding vertical for the wider India prediction engine (Part A).
It's the easiest place to prove the loop closes, because resolution data is
structured and unambiguous: a match either has a winner or it doesn't.

| Script | Data source | Purpose |
| --- | --- | --- |
| `cricket_engine.py` | Cricbuzz API (live) | the live loop — generate, lock, resolve, score |
| `backtesting.py` | Cricsheet archive (22,537 matches) | offline backtest and calibration study |

The backtest exists because live matches are sparse. Over a week you might
resolve a dozen markets — nowhere near enough to say anything about calibration.
Replaying the same question-generation logic over completed historical matches
gave about 700 scored predictions per model in an afternoon.

**Headline result.** The model's probabilities carry real signal (DeepSeek, 698
matches: 69.3% accuracy against about 50% for a coin flip, Brier 0.1918), and
they have a measurable calibration defect that a two-parameter recalibration
(Platt scaling) removes: out-of-fold expected calibration error (ECE) falls from
6.0 to 2.1 points, which is at the level that sampling noise alone produces.
Full write-up, with the figures and every correction made along the way, is in
`WRITEUP_PartB`.

---

## Part 1 — Live pipeline

```
fetch_matches()          Cricbuzz live + upcoming → filter to pre-match, high-engagement
        ↓
generate_question()      LLM → {question, probability, resolution_rule, reasoning}
        ↓
save_prediction()        SQLite, market_status = OPEN
        ↓
check_and_lock_markets() now >= lock_time (start − 15 min) → LOCKED
        ↓
check_and_resolve()      fetch_scorecard() → LLM resolver → YES / NO / UNRESOLVED
        ↓
                         RESOLVED (+ Brier score) or VOID
```

The live engine's LLM is DeepSeek (OpenAI-compatible API), at `temperature=0` so
that live runs are comparable with the backtest. The original run used
gpt-4o-mini.

### Design decisions

**Match selection.** A match qualifies only if it is genuinely pre-match (state
`Upcoming`/`Preview`, or status beginning "Match starts") *and* either the format
is `HUN` (The Hundred runs almost daily, giving a fast resolution cycle for
testing) or one side is a high-engagement international team (India, Australia,
England, Pakistan, South Africa, New Zealand, West Indies, Bangladesh).
`TBC`/`TBD` placeholders pass through so knockout fixtures aren't dropped early.

**Cache-busting.** Every Cricbuzz call appends `?t={epoch}`. Without it the
upstream cache serves stale states and a completed match keeps reporting as live,
silently stalling resolution.

**What the lock does and does not guarantee.** `lock_time = start − 15 min`,
which is *after* the toss (about 30 minutes before start), so the lock alone does
not keep toss information out. What keeps it out is that the probability is
generated when a fixture is first discovered, usually hours ahead, from series
name, format, the two teams and the venue only — no toss, innings or result.
Generating after the toss would need that prompt checked.

**Question scope in V1.** Restricted to match-winner questions, which resolve
from the final status string alone. Trades variety for a resolution path that
cannot be ambiguous.

**LLM resolution over string parsing.** `resolve_prediction()` is a few-shot
prompt returning exactly `YES`, `NO`, or `UNRESOLVED`. Cricbuzz status strings
vary widely ("won by 6 wickets", "match abandoned due to rain", "no result") and
a regex over them is brittle. The LLM reads rule and status together, with an
explicit escape hatch for abandonments, which become `VOID` rather than being
force-scored.

**Completion detection.** The API's `isMatchComplete` flag is unreliable — it
returns `False` on clearly finished matches. A match is treated as complete if
that flag is set *or* the status contains "won by", "abandoned", "draw", or
"no result".

**Self-scoring.** On resolution: `brier_score = (probability − actual)²` plus a
binary `is_correct` at the 0.5 threshold. Brier rewards calibration and
resolution together — a confident miss is punished hard.

### Schema

Single table `predictions` in `predictions.db`:

| Column | Notes |
| --- | --- |
| `id` | autoincrement |
| `matchId` | Cricbuzz id — dedupe key |
| `team1`, `team2`, `seriesName`, `match_format`, `venue` | fixture metadata |
| `match_date`, `match_start_time`, `lock_time` | epoch ms + ISO UTC |
| `market_status` | `OPEN` → `LOCKED` → `RESOLVED` \| `VOID` |
| `question`, `probability`, `resolution_rule` | the market |
| `probability_reasoning`, `reasoning` | basis for the number; why it's engaging |
| `actual_outcome`, `is_correct`, `brier_score` | settlement |
| `created_at` | ISO UTC |

`check_existing_prediction()` prevents duplicate markets across polling cycles.

### Demo — real run

A run on 2026-08-16 (original gpt-4o-mini version of the engine):

```
[Pass 1] Fetching & creating upcoming markets...
Found 2 target upcoming matches.
Created market #145005: Will Trent Rockets win this match?
Created market #145368: Will Trent Rockets Women win this match?
```

Market #145005 in full:

```json
{
  "question": "Will Trent Rockets win this match?",
  "probability": 0.55,
  "probability_reasoning": "Trent Rockets have shown strong performances in previous editions of The Hundred, and their squad depth appears to be slightly better than that of Manchester Super Giants. Additionally, playing at Lord's, a venue known for its balanced pitch, may favor their batting lineup. However, the Super Giants have also been competitive, making this a close call.",
  "resolution_rule": "The question will be resolved based on the final match result, specifically whether Trent Rockets are declared the winners.",
  "reasoning": "This question is compelling for fans as it directly involves the performance of a popular team in a prestigious venue. The narrative of The Hundred and the rivalry between these two teams adds excitement, making fans eager to see if their team can secure a victory."
}
```

That market lost and scored 0.3025, which is what an honest near-coin-flip looks
like. Only a handful of live markets have resolved so far, so no calibration
claim rests on live data.

Resolver checked against a result string: `"Will India win the match?"` +
`"India won by 6 wickets"` → `YES`.

---

## Part 2 — Backtest & calibration (Cricsheet)

### The experiment

For each sampled historical match:

1. **Parse** Cricsheet JSON → teams, format, venue, date, true winner. Matches
   with no high-engagement team are skipped.
2. **Blind the model.** The prompt gets only series name, format, teams and venue.
   The toss, innings and result are never included.
3. **Predict.** The model returns a binary question and P(team 1 wins) in
   0.01–0.99, at `temperature=0`.
4. **Score against reality.** Outcome from team 1's perspective: YES if team 1
   won, NO if they lost, VOID for draws/no-result (excluded). Brier and
   directional `is_correct` are stored per row.

Matches are drawn at random with a fixed seed, a match is never scored twice, and
each model's predictions live in their own SQLite file so recalibration
experiments reuse them without new LLM calls.

### Results

| Run | Matches | Brier (0.25 = always 50%) | Accuracy |
| --- | --- | --- | --- |
| gpt-4o-mini | 702 | 0.2018 | 70.8% |
| DeepSeek (replication, different sample) | 698 | 0.1918 (95% CI 0.1805–0.2033) | 69.3% (95% CI 65.6–72.8%) |

Accuracy is measured against about 50%, not against the 48.7% rate at which the
first-named team won: which team is "team 1" is arbitrary, and the best rule that
always picks one team scores 51.3%. DeepSeek by format: ODI 341 matches (Brier
0.2006, 67.4%), T20 242 (0.1923, 68.6%), Test 111 (0.1672, 76.6%); the Test
figure rests on only 111 matches. The two models score different random draws, so
their Brier scores are not a paired comparison.

### Calibration is the target

A user who takes a market priced at 30% accepts that risk for the matching
payout. If such events happen 20% of the time, the user is underpaid. So the
write-up treats **calibration (ECE and the gap in each confidence bin) as the
target** and Brier as a no-harm check. Brier ≈ uncertainty − resolution +
reliability, and recalibration can only reduce the small reliability term;
getting a materially better Brier needs better resolution, meaning more
information in the prompt.

Reliability by confidence bin (predicted % / actual win rate %, gap in points):

| Bin | gpt-4o-mini n | predicted / actual | gap | DeepSeek n | predicted / actual | gap |
| --- | --- | --- | --- | --- | --- | --- |
| 0–20% | 24 | 14.6 / 12.5 | −2.1 | 75 | 12.6 / 8.0 | −4.6 |
| 20–40% | 103 | 32.5 / 20.4 | **−12.1** | 83 | 33.6 / 21.7 | **−11.9** |
| 40–60% | 162 | 46.2 / 40.1 | −6.1 | 245 | 48.3 / 42.9 | −5.5 |
| 60–80% | 358 | 70.2 / 69.8 | −0.4 | 216 | 66.5 / 62.5 | −4.0 |
| 80–100% | 55 | 85.2 / 85.5 | +0.3 | 79 | 87.9 / 96.2 | +8.3 |

The 20–40% bin is the one that matters: both models say about 33% for team 1 and
team 1 wins about 21% of the time, a gap larger than sampling noise
(DeepSeek's 95% Wilson interval on 21.7% is 14.2–31.7%). Low and middle bins sit
above the actual win rates and DeepSeek's top bin sits below it. Platt's fitted
slope on DeepSeek is 1.32 (log-odds = −0.26 + 1.32 × model log-odds): the
probabilities are **too compressed toward 50%**, not too extreme. The top and
bottom bins are thin (75–79 matches for DeepSeek).

### Does recalibration fix it?

Tested out of fold: 40 repeats of stratified 5-fold cross-validation, so every
match is scored by a calibrator that never saw it, and the ECE is computed on the
pooled 698 out-of-fold predictions of each repeat. A single 60/40 split is one
draw and can mislead (on an earlier 418/280 split ECE went 5.3 → 4.4, whereas the
repeated out-of-fold result is 6.0 → 2.1).

ECE is binned (5 equal-width bins), so it has a noise floor: simulating outcomes
from the model's own probabilities, sampling noise alone produces an ECE of 2.8
points on average (4.6 at the 95th percentile) at 698 matches. ECE also can be
gamed by predicting the base rate, so it is read together with Brier.

| Method | ECE (points) | Worst bin gap (points) |
| --- | --- | --- |
| Raw | 6.0 | 11.9 |
| Platt | 2.1 | 4.7 |
| Isotonic | 1.7 | 3.9 |

Raw ECE is above the 95th percentile of noise, so the miscalibration is real; the
calibrated values are at or below the noise average, so they are
indistinguishable from perfect calibration. The 20–40% gap goes from −11.9 to
about −0.5 (Platt) and −0.2 (isotonic). Platt cannot fully fix the top bin
(+4.6 after Platt) while isotonic can (−1.4), but at 698 matches **Platt and
isotonic are not separable**: their ECE differs by 0.4 points, well inside the
noise.

Brier, as the no-harm check, improves by +0.0032 for Platt (95% CI −0.0007 to
+0.0069) and +0.0029 for isotonic: calibrating did no harm, and the gain is as
small as the decomposition predicts.

**How much calibration data is needed (learning curve).** Out-of-fold ECE when
the calibrator is fitted on only a random `size` of each training fold's ~558
matches (all 698 matches scored every time; 20 repeats):

| Calibrator trained on | 100 | 200 | 300 | 400 | 500 |
| --- | --- | --- | --- | --- | --- |
| Platt ECE | 3.2 | 2.6 | 2.2 | 2.2 | 2.2 |
| Isotonic ECE | 4.8 | 3.2 | 2.7 | 1.9 | 1.8 |

Raw ECE is 6.0 and the noise floor 2.8 throughout. Platt (two parameters) reaches
the floor with about 300 matches and stays flat; isotonic starts worse, needs
more data, and is still falling at 500, the most this sample allows.

**Decision.** Use Platt scaling now. Isotonic is adopted only if a larger sample
(about 1,500 matches or more) shows it better than Platt on out-of-fold ECE and
worst bin gap with a valid interval. The bootstrap interval the script currently
prints for the ECE reduction is biased (resampling adds noise that inflates the
calibrated ECE), so only the point estimates and the noise-floor comparison are
quoted.

### Is the team order a leak?

The prompt always names team 1 first, so the model could favour whichever team is
named first. Scoring DeepSeek in both orders: the mean probability for the
first-named team is 52.8% against 52.3% when the same team is named second, a
lean of about 0.5 points (95% CI −0.0 to +1.1, p = 0.052). The first-named team
actually wins 48.7% of the time (about 0.7 standard errors below 50%), so order
carries no information about the outcome. Averaging the two orders improves Brier
by only +0.0005 (CI −0.0017 to +0.0029), not significant.

### Skill or memorised results?

Cricket results are widely documented, so a model might recall outcomes from
pretraining. The test is to restrict to matches played after the model's training
cutoff. On 240 gpt-4o-mini matches dated after its cutoff, accuracy was 72.9%
(Wilson 95% CI about 67–78%) and Brier 0.1945, against 70.8% and 0.2018 on all
dates: the headline does not depend on recalled outcomes, but the sample is too
small to say post-cutoff is *better*, and equal accuracy before and after does
not prove the model never saw those matches. The DeepSeek post-cutoff run and a
direct recall probe have **not** been run yet.

---

## Setup

**Environment:** Python 3.11.

```bash
pip install -r requirements_cricket.txt   # openai, requests, numpy, pandas, scikit-learn, matplotlib
```

`sqlite3` is standard library. The backtest additionally needs the Cricsheet
JSON archive (on Kaggle: `cricksheet-data-22000-matches`).

**Credentials** are read from environment variables only — nothing is hard-coded:

```bash
export DEEPSEEK_API_KEY=...      # LLM (both scripts)
export RAPIDAPI_KEY=...          # RapidAPI → "Cricbuzz Cricket" (live engine only)
export DEEPSEEK_MODEL=...        # optional; default "deepseek-flash" (check DeepSeek's current model list)
```

> **Do not commit keys.** The earlier notebook version of the live loop contained
> a literal RapidAPI key. Rotate that key, and purge it from git history if the
> notebook was ever committed.

### Running

**Live engine**

```bash
python cricket_engine.py --selftest   # check the LLM resolver on 4 known cases (needs DEEPSEEK_API_KEY only)
python cricket_engine.py --once       # one pass (create / lock / resolve), then exit
python cricket_engine.py              # loop forever, one pass every 120 minutes (Ctrl+C to stop)
```

`--db` sets the SQLite file (default `predictions.db`) and `--interval-minutes`
the loop period.

**Backtest**

```bash
python backtesting.py --data-dir /path/to/cricsheet_json --n 250   # score up to 250 NEW matches, then report
python backtesting.py --eval-only                                  # report from the DB, no API calls
python backtesting.py --eval-only --calibration-experiment         # raw vs Platt vs isotonic, out-of-fold + learning curve
python backtesting.py --eval-only --calibration-experiment --prob-source avg   # same, on order-averaged probabilities
python backtesting.py --fill-swaps --n 250                         # add the swapped-order probability to matches already stored
python backtesting.py --swap-teams --n 250                         # score NEW matches in both orders (2 calls each)
python backtesting.py --after 2025-01-01 --db backtest_deepseek_postcutoff.db --n 250   # contamination test
```

Predictions go to `backtest_deepseek.db` by default (use a separate `--db` for any
filtered run so it is never mixed into another report). `--n` caps how many *new*
matches are scored per run to bound API cost. The experiment prints raw/Platt/
isotonic ECE, the worst bin gap, the per-bin gap table, the noise floor, Platt's
fitted parameters and the learning curve; it also saves `reliability.png`, a
reliability diagram with Wilson 95% intervals per bin.

---

## Limitations

**Live pipeline**

- **One question type, one market per fixture.** V1 generates match-winner
  questions only; dedupe is on `matchId`. Player milestones and in-play questions
  need the innings-level scorecard, which `fetch_scorecard()` already returns but
  the generator doesn't use.
- **Format filter is open.** `EXCLUDED_FORMATS` is empty, so a Test match can
  create a market that stays locked for up to five days.
- **Notebook-style scheduling.** The loop is a timer in one process; anything
  production-shaped needs a real scheduler and a database that survives restarts.
- **Resolution depends on an LLM call.** More flexible than parsing, but adds a
  dependency and slight non-determinism to settlement; `temperature=0` and a
  three-word output space bound the risk without removing it.
- **The calibrator is not wired into the live engine.** It stores raw
  probabilities and computes Brier at resolution. The next step is to fit Platt on
  backtest predictions and apply it as a separate step that keeps the raw number,
  then track ECE on live resolved markets as they accumulate.
- **Little live evidence.** Only a handful of live markets have resolved.

**Backtest**

- **No features reach the prompt.** Team form, squad and head-to-head data are
  absent. In one 250-prediction gpt-4o-mini run the model emitted only nine
  distinct probabilities (0.15 to 0.85); whether DeepSeek is equally coarse has
  not been measured.
- **Random, not chronological, order.** The matches are a random draw, and the
  out-of-fold tests shuffle them, so they are valid out-of-sample evaluation but
  not a time-series backtest and cannot detect drift across eras.
- **Binned ECE is noisy.** With five bins and about 700 matches, noise alone
  produces 2.8 points of ECE, so differences below that cannot be resolved (the
  floor falls roughly with √n, to about 1.7 at 1,900 matches).
- **No valid interval yet on the ECE reduction** (see Decision above).
- **Format-blind pooling.** Test, ODI and T20 share one calibrator even though
  Brier ranges from 0.167 (Test) to 0.201 (ODI) for DeepSeek. Per-format
  calibration needs more matches per format first.
- **Thin tails.** The top bins hold 55 (gpt-4o-mini) and 79 (DeepSeek)
  predictions, and an earlier 341-match run showed such bins can move a lot with
  more data.
- **Different samples per model**, so the two Brier scores are not a paired
  comparison.

**Both**

- **Probabilities are LLM priors, not a model.** They come from the model's
  general knowledge, with no feature input, which is why they cluster mid-range.
  Recalibration fixes the *gaps* but cannot add *resolution*.

### Next steps

1. **A feature-based prior** trained on Cricsheet ball-by-ball data (recent form,
   head-to-head, venue record) as the source of the number, or as an input beside
   the LLM's. The value is in resolution and granularity, not accuracy; a
   calibrator fitted today would need refitting on the new source.
2. **Isotonic against Platt on a larger sample:** score about 1,000 more matches
   (five batches of 250) and re-run the out-of-fold comparison, with a corrected
   interval on the ECE reduction.
3. **Put the Platt calibrator into the live engine** as a separate step.
4. **DeepSeek post-cutoff run and a recall probe**, then re-run the gpt-4o-mini
   test with a later cutoff.
5. Beta calibration, per-format calibration and bin weighting once the larger
   sample allows.