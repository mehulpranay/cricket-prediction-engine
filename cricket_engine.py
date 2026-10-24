"""
Cricket prediction engine (live loop) — DeepSeek version.

Pass 1  fetch upcoming matches from Cricbuzz, ask the LLM for a yes/no question + probability, save it OPEN
Pass 2  lock markets whose lock_time (match start - 35 min) has passed
Pass 3  for locked markets, read the final result from the scorecard, let the LLM resolve YES / NO / UNRESOLVED,
        and score it (Brier)


Run:
    python cricket_engine.py --selftest      # check the LLM resolver on 4 known cases (needs DEEPSEEK_API_KEY only)
    python cricket_engine.py --once          # one pass (create / lock / resolve), then exit
    python cricket_engine.py                 # loop forever, one pass every 120 minutes (Ctrl+C to stop)

Output format: DeepSeek has no json_schema response_format, so every LLM call is a forced, strict tool call
(the same pattern as the v2 entertainment engine). A Pydantic model defines the schema, the model must answer by
calling the tool, and the arguments are validated again in code. Strict mode is Beta-only, hence the /beta URL.
"""
import argparse
import os
import sqlite3
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Literal, Optional, Type, TypeVar

import requests
from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEEPSEEK_BASE_URL = "https://api.deepseek.com/beta"       # strict tool calls (schema-exact arguments) exist only on /beta
MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-flash")   # verify against DeepSeek's current model list
DB_PATH = "predictions.db"                                # override with --db
GENERATION_TEMPERATURE = 0.0   # was 0.7 in the OpenAI script; 0.0 matches the backtest, so live and backtest are comparable
RAPIDAPI_HOST = "cricbuzz-cricket.p.rapidapi.com"
DISCOVERY_MINUTES = 300        # --smart: fetch fixtures at most this often (the schedule is every 6h; 300 leaves room for scheduler jitter)
RESOLVE_CHECK_MINUTES = 300    # --smart: look at scorecards at most this often (same pass as discovery)
# --smart: no scorecard call for a market until this long after its start (a match cannot be over earlier), by Cricbuzz format
RESOLVE_AFTER_START_MINUTES = {"T20": 210, "HUN": 180, "ODI": 480}
RESOLVE_AFTER_DEFAULT_MINUTES = 360
MONTHLY_CALL_BUDGET = 190      # RapidAPI free plan is 200 calls a month; the engine stops calling at this count and waits for the next month
LOCK_MINUTES_BEFORE_START = 35   # before the toss (about 30 min before start), so toss information can never enter

HIGH_ENGAGEMENT_TEAMS = [
    "India", "Australia", "England", "Pakistan",
    "South Africa", "New Zealand", "West Indies", "Bangladesh", "TBC", "TBD"
]
EXCLUDED_FORMATS = ["TEST"]   # no Test matches: they last days and a draw has no YES/NO winner


# ---------------------------------------------------------------------------
# LLM + API plumbing
# ---------------------------------------------------------------------------

_client = None


def get_client() -> OpenAI:
    """DeepSeek speaks the OpenAI API, so the OpenAI SDK is used with a different base_url (the /beta one, for strict tool calls)."""
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url=DEEPSEEK_BASE_URL)
    return _client


# ---------------------------------------------------------------------------
# Structured output: forced strict tool calls
# ---------------------------------------------------------------------------

MARKET_TOOL_NAME = "submit_market"
RESOLUTION_TOOL_NAME = "submit_resolution"
TOOL_MAX_TOKENS = 1500     # doubled once if a reply is cut off mid-arguments
MAX_TOOL_ATTEMPTS = 2


# Plain comments, not docstrings: Pydantic would copy a docstring into the schema sent to the model.
# Field order is generation order, and it matches the order of the old JSON format.
# No defaults anywhere: strict mode wants every field required.
class Market(BaseModel):
    model_config = ConfigDict(extra="forbid")   # additionalProperties: false, which strict mode requires

    question: str = Field(description='One yes/no question: will the first-named team win this match, e.g. "Will Southern Brave win this match against Sunrisers Leeds?"')
    probability: float = Field(ge=0.0, le=1.0, description="Your probability, between 0.01 and 0.99, that the question resolves YES (that the first-named team wins).")
    probability_reasoning: str = Field(description="Explain specifically what historical knowledge, team strength, venue history, or contextual factors informed your pre-match probability estimate. Be honest if you have limited information.")
    resolution_rule: str = Field(description='How the question resolves from the final result status string, e.g. "Resolves YES if India wins, NO if they lose or draw".')
    reasoning: str = Field(description="Explain why this question is compelling and significant for fans: what narrative, rivalry, or stakes make it worth predicting on.")

    # minLength is not allowed in strict-mode schemas, so blank strings are rejected here instead
    @field_validator("question", "resolution_rule")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("must not be blank")
        return v


class Resolution(BaseModel):
    model_config = ConfigDict(extra="forbid")

    outcome: Literal["YES", "NO", "UNRESOLVED"] = Field(description="YES or NO if the match result settles the question; UNRESOLVED if the match was abandoned or the result is unclear.")


_STRIP_KEYS = {"title", "default"}   # keywords Pydantic emits that the strict validator may reject


def _clean_schema(node: Any) -> Any:
    """Recursively drops keywords the strict-mode validator may not accept. Property names are never stripped."""
    if isinstance(node, dict):
        out = {}
        for k, v in node.items():
            if k in _STRIP_KEYS:
                continue
            if k == "properties" and isinstance(v, dict):
                out[k] = {name: _clean_schema(sub) for name, sub in v.items()}
            else:
                out[k] = _clean_schema(v)
        return out
    if isinstance(node, list):
        return [_clean_schema(v) for v in node]
    return node


def _tool_spec(name: str, description: str, model_cls: Type[BaseModel]) -> dict:
    # Not a real function: a structured-output channel. We never execute it, we just read its arguments.
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "strict": True,
            "parameters": _clean_schema(model_cls.model_json_schema()),
        },
    }


MARKET_TOOL_SPEC = _tool_spec(MARKET_TOOL_NAME, "Submit the prediction market for this match. Call this exactly once.", Market)
RESOLUTION_TOOL_SPEC = _tool_spec(RESOLUTION_TOOL_NAME, "Submit how this prediction resolved. Call this exactly once.", Resolution)

M = TypeVar("M", bound=BaseModel)


def _call_tool(prompt: str, temperature: float, tool_spec: dict, model_cls: Type[M]) -> Optional[M]:
    """One LLM call that must answer through the given strict tool, validated into a Pydantic object.
    Strict mode guarantees the shape; the validators cover meaning. Retries only for the rare cases:
    truncation, no tool call, validation failure. Returns None if every attempt fails.
    Thinking stays off (reasoning_effort='none'): forced tool choice is rejected in thinking mode, and it keeps
    temperature in force."""
    name = tool_spec["function"]["name"]
    messages = [{"role": "user", "content": prompt}]
    max_tokens = TOOL_MAX_TOKENS

    for attempt in range(1, MAX_TOOL_ATTEMPTS + 1):
        try:
            resp = get_client().chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=temperature,
                tools=[tool_spec],
                tool_choice={"type": "function", "function": {"name": name}},
                reasoning_effort="none",
                max_tokens=max_tokens,
            )
        except Exception as e:
            print(f"LLM call failed (attempt {attempt}): {e}")
            continue

        choice = resp.choices[0]
        msg = choice.message

        # Case 1: truncated mid-arguments -> raise the limit, don't just repeat.
        if choice.finish_reason == "length":
            max_tokens *= 2
            print(f"{name}: reply truncated (attempt {attempt}); retrying with max_tokens={max_tokens}")
            continue

        # Case 2: the model answered in plain text instead of calling the tool.
        if not msg.tool_calls:
            print(f"{name}: no tool call returned (attempt {attempt}); nudging.")
            messages.append({"role": "assistant", "content": msg.content or "(no tool call made)"})
            messages.append({"role": "user", "content": f"You must call {name}."})
            continue

        # Case 3: tool call present -> validate.
        call = msg.tool_calls[0]
        try:
            return model_cls.model_validate_json(call.function.arguments)
        except ValidationError as e:
            print(f"{name}: validation failed (attempt {attempt}): {e}")
            messages.append(msg)
            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": f"Validation failed: {e}. Call {name} again with a corrected payload.",
            })

    return None


def _spend_call() -> bool:
    """Counts one RapidAPI call against this month's budget (kept in the DB, so it survives between runs).
    Returns False, and counts nothing, once the budget is used up."""
    key = "calls_" + datetime.now(timezone.utc).strftime("%Y-%m")
    used = int(_meta_get(key) or 0)
    if used >= MONTHLY_CALL_BUDGET:
        print(f"RapidAPI budget for this month used up ({used}/{MONTHLY_CALL_BUDGET}); skipping the call.")
        return False
    _meta_set(key, str(used + 1))
    return True


def _rapid_headers() -> dict:
    key = os.environ.get("RAPID_API_KEY")
    if not key:
        raise RuntimeError("Set the RAPID_API_KEY environment variable (RapidAPI -> Cricbuzz Cricket).")
    return {"x-rapidapi-key": key, "x-rapidapi-host": RAPIDAPI_HOST}


# ---------------------------------------------------------------------------
# Pass 1 inputs: fixtures + question generation
# ---------------------------------------------------------------------------

def fetch_matches():
    # t=<epoch> busts Cricbuzz's upstream cache; without it a finished match can keep reporting as live
    url_upcoming = f"https://{RAPIDAPI_HOST}/matches/v1/upcoming?t={int(time.time())}"

    headers = _rapid_headers()

    seen_match_ids = set()
    target_matches = []

    for url in [url_upcoming]:   # the live endpoint only lists matches already in progress, which are skipped below anyway

        if not _spend_call():
            continue
        try:
            response = requests.get(url, headers=headers, timeout=20)
            response.raise_for_status()

            data = response.json()

            # Check for RapidAPI quota or error messages
            if isinstance(data, dict) and "message" in data:
                print(f"API Error Message: {data['message']}")
                continue

            for type_match in data.get("typeMatches", []):
                for series in type_match.get("seriesMatches", []):
                    wrapper = series.get("seriesAdWrapper")
                    matches_list = wrapper.get("matches", []) if wrapper else series.get("matches", [])

                    for match in matches_list:
                        info = match.get("matchInfo")
                        if not info:
                            continue

                        match_id = info["matchId"]
                        if match_id in seen_match_ids:
                            continue

                        team1 = info.get("team1", {}).get("teamName", "TBC")
                        team2 = info.get("team2", {}).get("teamName", "TBC")
                        series_name = info.get("seriesName", "Unknown Series")
                        fmt = info.get("matchFormat", "")
                        state = info.get("state", "").title()
                        status = info.get("status", "")

                        # Ensure match has not started yet
                        is_pre_match = state in ["Upcoming", "Preview", ""] or status.startswith("Match starts")
                        is_in_progress = state in ["In Progress", "Inning Break", "Complete"] or "opt to" in status.lower() or "need" in status.lower()

                        if is_in_progress or not is_pre_match:
                            continue

                        # EXACTLY TWO CONDITIONS: Format is HUN OR Team is High Engagement
                        is_hun = (fmt == "HUN")
                        is_high_engagement = (team1 in HIGH_ENGAGEMENT_TEAMS or team2 in HIGH_ENGAGEMENT_TEAMS)

                        if fmt not in EXCLUDED_FORMATS and (is_hun or is_high_engagement):
                            start_time_sec = int(info["startDate"]) / 1000
                            lock_time_sec = start_time_sec - (LOCK_MINUTES_BEFORE_START * 60)

                            start_iso = datetime.fromtimestamp(start_time_sec, tz=timezone.utc).isoformat()
                            lock_iso = datetime.fromtimestamp(lock_time_sec, tz=timezone.utc).isoformat()

                            target_matches.append({
                                "matchId": match_id,
                                "seriesName": series_name,
                                "matchFormat": fmt,
                                "team1": team1,
                                "team2": team2,
                                "state": state if state else "Preview",
                                "status": status,
                                "startDate": info["startDate"],
                                "startTimeIso": start_iso,
                                "lockTimeIso": lock_iso,
                                "venue": info.get("venueInfo", {}).get("ground", "Unknown Ground")
                            })
                            seen_match_ids.add(match_id)
        except requests.exceptions.HTTPError as err:
            print(f"HTTP Error occurred for {url}: {err}")
        except Exception as e:
            print(f"Unexpected error for {url}: {e}")

    return target_matches


def fetch_scorecard(match_id):
    """
    Fetches the scorecard for a specific cricket match from Cricbuzz API.

    Input:
        - match_id : unique Cricbuzz match identifier (from fetch_matches output)

    Process:
        - Calls mcenter/v1/{match_id}/hscard endpoint
        - Adds cache-busting timestamp parameter to ensure fresh data every call
        - Extracts match completion status, result string, and innings details

    Note:
    - isMatchComplete flag is unreliable — Cricbuzz sometimes returns False even
      when match is complete. Use "won by" or "abandoned" in status string as
      primary completion check.

    Returns None if the request fails, so one bad call never stops the other markets from being checked.
    """
    url = f"https://{RAPIDAPI_HOST}/mcenter/v1/{match_id}/hscard?t={int(time.time())}"
    if not _spend_call():
        return None

    try:
        response = requests.get(url, headers=_rapid_headers(), timeout=20)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        print(f"Scorecard fetch failed for match {match_id}: {e}")
        return None

    return {
        "isMatchComplete": data.get("isMatchComplete", False),
        "status": data.get("status", ""),
        "scorecard": data.get("scorecard", [])
    }


def generate_question(match_info):
    team1 = match_info['team1']
    team2 = match_info['team2']
    # Same wording and the same Match Context block as backtest.py's prompt, so the Platt calibrator fitted on the backtest
    # applies to the live probability. Only the output instructions differ (a tool call with the extra text fields).
    # Team 1 is always the question's subject, which keeps the probability's meaning fixed ("team 1 wins").
    prompt = f"""
You are an expert cricket analyst and prediction market maker. 
Analyze the match context and submit your market prediction.

Match Context:
- Series: {match_info['seriesName']} ({match_info['matchFormat']})
- Teams: {team1} vs {team2}
- Venue: {match_info['venue']}

You must submit EXACTLY these fields:
- "question": A binary question framed as "Will {team1} win this match against {team2}?"
- "probability": A float between 0.01 and 0.99 representing the estimated probability that {team1} wins.
- "probability_reasoning", "resolution_rule" and "reasoning": see the field descriptions.

Submit your answer by calling the function {MARKET_TOOL_NAME}.
"""

    market = _call_tool(prompt, GENERATION_TEMPERATURE, MARKET_TOOL_SPEC, Market)
    if market is None:
        return None

    return {
        "question": market.question,
        "probability": min(max(market.probability, 0.01), 0.99),   # same 0.01-0.99 range the backtest allows
        "probability_reasoning": market.probability_reasoning,
        "resolution_rule": market.resolution_rule,
        "reasoning": market.reasoning,
    }


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            matchId TEXT,
            team1 TEXT,
            team2 TEXT,
            seriesName TEXT,
            match_format TEXT,
            match_date TEXT,
            match_start_time TEXT,
            lock_time TEXT,
            market_status TEXT DEFAULT 'OPEN',
            venue TEXT,
            question TEXT,
            probability REAL,
            probability_reasoning TEXT,
            resolution_rule TEXT,
            reasoning TEXT,
            actual_outcome TEXT,
            is_correct INTEGER,
            created_at TEXT,
            brier_score REAL
        )
    ''')
    cursor.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)")
    conn.commit()
    conn.close()


def _meta_get(key):
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    conn.close()
    return row[0] if row else None


def _meta_set(key, value):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("INSERT INTO meta (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value", (key, value))
    conn.commit()
    conn.close()


def _due(key, every_minutes):
    """True if `key` was never stamped, or was stamped more than `every_minutes` ago."""
    last = _meta_get(key)
    if not last:
        return True
    return (datetime.now(timezone.utc) - datetime.fromisoformat(last)).total_seconds() >= every_minutes * 60


def save_prediction(match_info, question_data):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO predictions 
        (matchId, team1, team2, seriesName, match_format, match_date, 
        match_start_time, lock_time, market_status, venue,
        question, probability, probability_reasoning, resolution_rule, reasoning, 
        actual_outcome, is_correct, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'OPEN', ?, ?, ?, ?, ?, ?, NULL, NULL, ?)
    ''', (
        match_info['matchId'],
        match_info['team1'],
        match_info['team2'],
        match_info['seriesName'],
        match_info['matchFormat'],
        match_info['startDate'],
        match_info['startTimeIso'],
        match_info['lockTimeIso'],
        match_info.get('venue', 'Unknown'),
        question_data['question'],
        question_data['probability'],
        question_data['probability_reasoning'],
        question_data['resolution_rule'],
        question_data['reasoning'],
        datetime.now(timezone.utc).isoformat()
    ))
    conn.commit()
    conn.close()


def check_existing_prediction(match_id):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT id FROM predictions WHERE matchId = ?', (str(match_id),))
    result = cursor.fetchone()
    conn.close()
    return result


# ---------------------------------------------------------------------------
# Pass 3: resolution
# ---------------------------------------------------------------------------

def resolve_prediction(prediction, status_string):
    prompt = f"""
You are resolving a cricket prediction based on match result.

Based on the match status, did this question resolve YES or NO?
If match was abandoned or result is unclear, say UNRESOLVED.

Submit your answer by calling the function {RESOLUTION_TOOL_NAME} with outcome YES, NO, or UNRESOLVED.

Examples:

Question: "Will India win the match?"
Resolution rule: "Resolves YES if India wins, NO if they lose or draw"
Match status: "India won by 6 wickets"
Answer: YES

Question: "Will Australia win the match?"
Resolution rule: "Resolves YES if Australia wins, NO if they lose or draw"
Match status: "England won by 45 runs"
Answer: NO

Question: "Will Sunrisers Leeds win the match?"
Resolution rule: "Resolves YES if Sunrisers Leeds wins, NO if they lose"
Match status: "Match abandoned due to rain"
Answer: UNRESOLVED

Question: {prediction['question']}
Resolution rule: {prediction['resolution_rule']}
Match status: {status_string}
"""
    result = _call_tool(prompt, 0.0, RESOLUTION_TOOL_SPEC, Resolution)
    if result is None:
        # the caller (check_and_resolve) catches this, leaves the market LOCKED and tries again next pass
        raise RuntimeError("resolver returned no valid tool call")
    return result.outcome


def selftest_resolver():
    """Checks the LLM resolver on four known cases. Needs DEEPSEEK_API_KEY only."""
    cases = [
        ("Will India win the match?", "Resolves YES if India wins, NO if they lose or draw", "India won by 6 wickets", "YES"),
        ("Will Australia win the match?", "Resolves YES if Australia wins, NO if they lose or draw", "England won by 45 runs", "NO"),
        ("Will Sunrisers Leeds win the match?", "Resolves YES if Sunrisers Leeds wins, NO if they lose", "Match abandoned due to rain", "UNRESOLVED"),
        ("Will India lose this match?", "Resolves YES if India loses, NO if they win or draw", "India won by 6 wickets", "NO"),
    ]
    ok = 0
    for question, rule, status, expected in cases:
        try:
            got = resolve_prediction({"question": question, "resolution_rule": rule}, status)
        except Exception as e:
            got = f"ERROR ({e})"
        passed = got == expected
        ok += passed
        print(f"{'PASS' if passed else 'FAIL'} | {question!r} + {status!r} -> {got} (expected {expected})")
    print(f"{ok}/{len(cases)} passed with model '{MODEL}'.")
    return ok == len(cases)


def check_and_lock_markets():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute('''
        SELECT id, question, lock_time 
        FROM predictions 
        WHERE market_status = 'OPEN'
    ''')
    open_markets = cursor.fetchall()

    now_utc = datetime.now(timezone.utc)

    for market in open_markets:
        market_id, question, lock_time_iso = market
        lock_dt = datetime.fromisoformat(lock_time_iso)

        if now_utc >= lock_dt:
            cursor.execute('''
                UPDATE predictions 
                SET market_status = 'LOCKED' 
                WHERE id = ?
            ''', (market_id,))
            print(f"🔒 Locked market #{market_id}: {question}")

    conn.commit()
    conn.close()


def check_and_resolve(smart=False):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute('''
        SELECT id, matchId, question, resolution_rule, probability, match_start_time, match_format
        FROM predictions 
        WHERE market_status = 'LOCKED'
    ''')
    locked_predictions = cursor.fetchall()

    if smart:
        # Scorecard calls cost quota: skip markets whose match cannot be over yet, and look at most once per interval.
        now = datetime.now(timezone.utc)
        def _old_enough(p):
            wait = RESOLVE_AFTER_START_MINUTES.get((p[6] or "").upper(), RESOLVE_AFTER_DEFAULT_MINUTES)
            return datetime.fromisoformat(p[5]) <= now - timedelta(minutes=wait)
        locked_predictions = [p for p in locked_predictions if _old_enough(p)]
        if not locked_predictions or not _due("last_resolve_check", RESOLVE_CHECK_MINUTES):
            conn.close()
            print("Nothing to resolve right now (no scorecard calls made).")
            return
        _meta_set("last_resolve_check", now.isoformat())

    for pred in locked_predictions:
        pred_id, match_id, question, resolution_rule, probability = pred[:5]

        scorecard = fetch_scorecard(match_id)
        if scorecard is None:
            print(f"⏳ Could not read the scorecard for #{pred_id}; will retry next pass.")
            continue
        status = scorecard['status']

        is_complete = (
            scorecard['isMatchComplete'] or
            "won by" in status.lower() or
            "abandoned" in status.lower() or
            "draw" in status.lower() or
            "no result" in status.lower()
        )

        if is_complete:
            try:
                outcome = resolve_prediction(
                    {'question': question, 'resolution_rule': resolution_rule},
                    status
                )
            except Exception as e:
                print(f"⏳ LLM resolution failed for #{pred_id} ({e}); will retry next pass.")
                continue

            if outcome in ['YES', 'NO']:
                actual_numeric = 1 if outcome == "YES" else 0
                is_correct = 1 if (
                    (outcome == 'YES' and probability >= 0.5) or
                    (outcome == 'NO' and probability < 0.5)
                ) else 0

                brier_score = (probability - actual_numeric) ** 2

                cursor.execute('''
                    UPDATE predictions 
                    SET actual_outcome = ?, is_correct = ?, brier_score = ?, market_status = 'RESOLVED'
                    WHERE id = ?
                ''', (outcome, is_correct, brier_score, pred_id))

                print(f"✅ RESOLVED #{pred_id}: {question} → {outcome} | Brier: {brier_score:.4f}")

            elif outcome == 'UNRESOLVED':
                cursor.execute('''
                    UPDATE predictions 
                    SET actual_outcome = 'VOID', market_status = 'VOID'
                    WHERE id = ?
                ''', (pred_id,))

                print(f"⚠️ VOIDED #{pred_id}: Match abandoned or no result.")

            else:
                print(f"⚠️ Unexpected resolver output {outcome!r} for #{pred_id}; leaving it LOCKED.")
        else:
            print(f"⏳ Still in progress (#{pred_id}): {status}")

    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# The three passes
# ---------------------------------------------------------------------------

def run_pipeline(smart=False):
    print(f"\n{'='*50}")
    print(f"Pipeline run: {datetime.now(timezone.utc).isoformat()}")
    print(f"{'='*50}")

    # -------------------------------------------------------------
    # Pass 1: Fetch Upcoming Matches & Create Open Markets
    # -------------------------------------------------------------
    print("\n[Pass 1] Fetching & creating upcoming markets...")
    if smart and not _due("last_discovery", DISCOVERY_MINUTES):
        upcoming_matches = []
        print("Fixture list was fetched recently; skipping (no API calls made).")
    else:
        upcoming_matches = fetch_matches()
        _meta_set("last_discovery", datetime.now(timezone.utc).isoformat())
        print(f"Found {len(upcoming_matches)} target upcoming matches.")

    for match in upcoming_matches:
        existing = check_existing_prediction(match['matchId'])
        if not existing:
            print(f"Generating market for: {match['team1']} vs {match['team2']} ({match['seriesName']})")
            question_data = generate_question(match)
            if question_data:
                save_prediction(match, question_data)
                print(f"Created market #{match['matchId']}: {question_data['question']}")
        else:
            print(f"Market already exists for matchId: {match['matchId']}")

    # -------------------------------------------------------------
    # Pass 2: Lock Expired Open Markets (Time >= Lock Time)
    # -------------------------------------------------------------
    print("\n[Pass 2] Checking open markets for betting lock...")
    check_and_lock_markets()

    # -------------------------------------------------------------
    # Pass 3: Check & Resolve Locked Markets
    # -------------------------------------------------------------
    print("\n[Pass 3] Checking locked markets for resolution...")
    check_and_resolve(smart)


def main():
    global DB_PATH

    ap = argparse.ArgumentParser(description="Cricket prediction engine (DeepSeek).")
    ap.add_argument("--once", action="store_true", help="run one pass (create / lock / resolve) and exit")
    ap.add_argument("--smart", action="store_true", help="with --once: make API calls only when something is due (for scheduled runs)")
    ap.add_argument("--selftest", action="store_true", help="check the LLM resolver on four known cases and exit")
    ap.add_argument("--interval-minutes", type=int, default=120, help="minutes between passes in loop mode")
    ap.add_argument("--db", default=DB_PATH, help="SQLite file (default: predictions.db)")
    args = ap.parse_args()
    DB_PATH = args.db

    if args.selftest:
        raise SystemExit(0 if selftest_resolver() else 1)

    init_db()

    if args.once:
        run_pipeline(args.smart)
        return

    try:
        while True:
            try:
                run_pipeline()
            except Exception as e:   # a transient API/network error must not kill a long-running loop
                print(f"Pass failed: {e!r}; will retry next cycle.")
            print(f"\nSleeping for {args.interval_minutes} minutes...")
            time.sleep(args.interval_minutes * 60)
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()