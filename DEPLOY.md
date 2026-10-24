# Running the live engine on GitHub Actions + GitHub Pages

What runs: every 6 hours `.github/workflows/cricket-engine.yml` runs `cricket_engine.py --once --smart`, rebuilds `docs/index.html`
with `build_site.py`, and commits `data/predictions.db` and `docs/` if they changed. One run does all three passes: it fetches the
upcoming fixtures and creates markets (1 RapidAPI call), locks any market past its lock time (no call), and fetches the scorecard of each
locked market that can be finished by now (1 call each). A market whose match has not started yet needs no call, and the engine waits
after the start before asking for a result: 3.5 h for T20, 3 h for The Hundred, 8 h for ODI. Test matches are excluded (`EXCLUDED_FORMATS`).

RapidAPI budget (200 calls a month on the free plan): 4 fixture calls a day is about 120 a month, and 10 to 15 matches a month need roughly 15 to 25 scorecard calls, so about 145 in all.
The engine keeps a counter in the database and stops calling at 190 for the month. If there are many matches a day, set `DISCOVERY_MINUTES`
higher in `cricket_engine.py` and change the cron to run less often.

Because runs are 6 hours apart, a market is marked LOCKED up to 6 hours after its lock time. That does not change the prediction (the
probability is saved when the market is created); the page itself closes guessing at `lock_time`.

## One-time setup
1. Put these files in a new GitHub repo: `cricket_engine.py`, `build_site.py`, `requirements_live.txt`, `.github/workflows/cricket-engine.yml`.
   Make the repo **public** (Actions minutes are unlimited for public repos; a private repo gets 2,000 free minutes a month,).
2. Repo -> Settings -> Secrets and variables -> Actions -> New repository secret: add `DEEPSEEK_API_KEY` and `RAPIDAPI_KEY` (the NEW RapidAPI key; rotate the old one first).
3. Settings -> Actions -> General -> Workflow permissions -> "Read and write permissions".
4. Run the workflow once by hand: Actions tab -> cricket-engine -> Run workflow. It creates `data/predictions.db` and `docs/`.
5. Settings -> Pages -> Source: "Deploy from a branch", branch `main`, folder `/docs`. The page appears at `https://<user>.github.io/<repo>/`.

## Things to know
- GitHub disables scheduled workflows after 60 days without repo activity. The bot's commits may not count, so check the Actions tab now and then (or push a small change).
- Never put keys in code or in the repo. The old key in the first version of `cricket_engine.py` should be rotated.
- The workflow's `concurrency` block makes overlapping runs wait their turn, so two runs never write the database together.
- Test locally first: `python cricket_engine.py --once --smart --db data/predictions.db` then `python build_site.py --db data/predictions.db --out docs` and open `docs/index.html`.