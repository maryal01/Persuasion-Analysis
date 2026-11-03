# AGENTS.md

Shared instructions for everyone (and every coding agent) working in this repository. Personal or machine-specific settings (local data paths, local environments, personal preferences) stay out of this file; use an untracked `AGENTS.local.md` / `CLAUDE.local.md`.

## Project

CS577 (NLP) course project **"Who Changes Their Mind?"**: predict persuasion from the arguments and a description of the listener, on Reddit r/changemyview (Webis-CMV-20) and debate.org (DDO).

Research questions: (1) can we predict whether a listener changes their mind, given arguments and a listener description; (2) how much comes from the text vs. the listener, and does it transfer between platforms; (3) stretch: do predictions match real audiences on unseen debates.

| Task | Definition | Status |
|---|---|---|
| T1 CMV persuasion | score both replies of a (delta, no-delta) pair; the delta reply should score higher | first run done (ModernBERT-base, text only, pairwise acc 66.1%, one seed; `RESULTS.md`) |
| T2 DDO persuasion | decided voter: did they switch; tied voter: Pro/Con/Tie after | datasets built, not trained |
| T3 Audience profiling | predict a person's stance on broad dimensions from their activity | not started |
| T4 DDO vote prediction | reproduce Durmus & Cardie's "more convincing" vote prediction | not started |
| T5 CMV malleability | from the OP's post and history, predict whether they award any delta | not started; Webis-CMV-20 has no raw history text and the history is far longer than 8,192 tokens |

Training runs on Purdue RCAC Gilbreth (`train.sub`). Hugging Face org `nlp-final-project11` holds the datasets (`persuasion-datasets`, branch `t1-t2`) and private models (`nlp-project-modernbert-<config>`).

## Where to look first

`main` can be stale and work lives on branches. Before saying a task is done or not: `git fetch origin && git branch -a -vv && git log --all --oneline --graph | head -30`, then read `README.md` and `RESULTS.md`. The table above can be out of date; update it when it changes.

## Structure

```
README.md  RESULTS.md  requirements.txt  AGENTS.md  CLAUDE.md
scripts/      commands you run, one folder per task: t1_t2/, t3/, t4/, t5/ (+ common/ for code shared by tasks)
jobs/         Slurm files (train.sub)
analysis/     one folder per analysis: <YYYY-MM-DD>_<name>/ with a README.md
notebooks/    exploration only
```

- No loose scripts in the repo root; put new files in one of the folders above.
- Scripts live in the folder of their task (`scripts/t5/build_dataset.py`, `scripts/t5/train.py`, ...); code used by several tasks goes to `scripts/common/` instead of being copied. The T1/T2 scripts (`create_dataset.py`, `train_delta.py`) still sit directly in `scripts/` and move to `scripts/t1_t2/` in a separate PR after the structure PR is merged.
- Within a task, the dataset is a CLI argument (`--dataset-config`), not a copied script. Splits are grouped (by post / debate; by person for T3/T5) and use one shared split function.
- Add a `src/` package and `tests/` once code is shared between several scripts.
- An analysis README has a Status line and Goal, What it does, Run (exact command), Results, Notes. Outputs and data stay out of git. Say so when a statistic comes from a sample.
- Results go into `RESULTS.md` (one row per finished run: numbers, seed, model repo, job id).

## Git

- Start from an up-to-date `main` (`git pull origin main`). **New piece of work = new branch**; never commit directly to `main`. Name it after the work (`audience-dimensions`, `repo-structure`). Keep one purpose per branch.
- Merge through pull requests. Put the task in the PR title (`T3: audience dimensions and activity cut-off`) and link the task's issue (`Refs #3`, `Closes #3`). Branch names are temporary; PRs and issues are the history.
- A merged branch is finished; follow-up work starts a new branch from `main`.
- New or rebuilt Hugging Face dataset configs go to a Hub branch (e.g. `t1-t2`), not the dataset repo's `main`.
- Never commit data, checkpoints, caches or tokens. Read tokens from the environment (`HF_TOKEN`). Never force-push shared branches.

## Gilbreth (RCAC)

Environment at `$RCAC_SCRATCH/persuasion-env`; submit with `sbatch -A <account> train.sub <config>` from a clone on scratch (setup in the header of `train.sub`). Heavy work only through `sbatch`, not on login nodes. Scratch is purged after 60 days: keep results elsewhere. Run `--smoke` before a long job.

## Code and reporting

- Scripts start with a docstring (what it does, usage example), use type hints and descriptive names, take paths as CLI arguments and read large files as streams.
- State assumptions you have not verified, with the results that depend on them. Do not hard-code values a dataset or library defines (label names, class order, split sizes). Do not hide negative results.
