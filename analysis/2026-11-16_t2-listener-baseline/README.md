# T2 listener-only baselines for `ddo_switch` and `ddo_undecided`

**Status:** done for `ddo_switch` (2026-11-16) and for `ddo_undecided` (2026-11-17). The sections
from Goal to Notes are about `ddo_switch`; `ddo_undecided` has its own section at the end.

## Goal

How well can we predict whether a decided voter switches sides from the voter alone, without the
debate text? This is the "listener" side of RQ2. The number to set it against is the text-only
ModernBERT on the same test rows (AUROC 0.719).

## What it does

`scripts/t1_t2/listener_baseline.py` fits class-weighted logistic regressions on the rows and splits
of the Hub dataset `ddo_switch` (branch `main`). It reads only the id and label columns, so the
transcripts are not downloaded. C is chosen on validation AUROC and the test split is scored once per
model. Feature groups:

- **Side before × year:** the voter's side before the debate and the debate's year. Not about the
  listener; it is the strongest no-text row in `RESULTS.md`.
- **Profile:** the voter's self-description in `users.json`: political and religious ideology, party,
  preferred president, gender, education, ethnicity, income and the 48 "big issues", one-hot encoded.
- **Voting record:** three numbers from the voter's votes in training debates: log(1 + switches),
  log(1 + stays) and a flag for "no votes". A training row never counts its own debate.
- **Voting record, earlier years only:** the same, counting only training debates from earlier years
  than the row's debate.

Two extra numbers:

- **Same-debate pairs:** among voters of one debate with the same side before, one of whom switched
  and one stayed, how often the switcher gets the higher score. A text-only model gives both voters
  the same input, so it scores 50% by construction.
- **Profile, trained on unseen voters only:** the profile model trained only on voters who never
  appear in validation or test, so it cannot recognise a frequent voter by their answers.

## Run

```bash
# ddo_switch (the default config)
python scripts/t1_t2/listener_baseline.py --users ddo/users.json --output outputs/listener_baseline.json

# ddo_undecided
python scripts/t1_t2/listener_baseline.py --users ddo/users.json --dataset-config ddo_undecided \
    --output outputs/listener_baseline_undecided.json
```

Needs scikit-learn (added to `requirements.txt`) and `users.json` from the DDO corpus. Each command
takes about a minute on a laptop. The script reads the class names from the dataset and picks the
metric from the number of classes: AUROC for two, macro-F1 and accuracy for more.

## Results

Test set: 5,845 votes from 2,504 debates, 368 switched. These are the rows the text model is scored
on. The interval is a bootstrap over debates (1,000 resamples).

| Model | Features | Test AUROC | 95% interval | Same-debate pairs |
|---|---|---|---|---|
| Side before × year | 38 | 0.643 | 0.612–0.671 | 50.0% |
| Profile | 419 | 0.678 | 0.650–0.706 | 63.7% |
| Profile, trained on unseen voters only | 419 | 0.606 | 0.577–0.636 | 52.1% |
| Voting record, earlier years only | 3 | 0.674 | 0.646–0.700 | 62.9% |
| **Voting record** | 3 | **0.790** | 0.765–0.813 | 78.3% |
| Profile + voting record | 422 | 0.790 | 0.763–0.815 | 77.5% |
| Profile + voting record + side × year | 460 | 0.805 | 0.780–0.829 | 78.2% |
| ModernBERT-base, text only (`RESULTS.md`) | – | 0.719 | about 0.69–0.75 | 50.0% by construction |

- **The voter's voting record alone scores higher than the debate text** (0.790 against 0.719), with
  three features. Whether a voter switches depends more on who the voter is than on the debate.
- **The profile adds nothing once the voting record is known** (0.790 with and without it).
- **Within a debate the record tells voters apart.** Of 801 same-debate pairs, it ranks the switcher
  higher in 78.3%.

Coverage: 83.8% of test rows have a profile record, 97.3% have a voter who also votes in a training
debate, and 60.7% have a voter with training votes from an earlier year.

## Notes

- **The voting record uses later votes too.** It counts all training debates, including ones after
  the row's debate. With earlier years only it scores 0.674, partly because 39% of test rows then have
  no record at all. The dataset has the debate's year and no finer date.
- **The profile model partly recognises frequent voters.** Trained on unseen voters only, it scores
  0.606 and ranks same-debate pairs at 52%. That model also has far less training data (5,615 of
  48,479 rows), so the two effects can't be separated here.
- **The profile is a snapshot from the crawl,** taken after the debates.
- **The comparison with the text model is unpaired.** The text model's test scores were not saved, so
  its interval is the rough ±0.03 from `RESULTS.md`, and the two models have not been combined yet.
- **Not in `RESULTS.md` yet.** The T2 section is still in PR #25; the rows above go there once it
  merges.
- The script reads only the profile fields listed above from `users.json`.

## `ddo_undecided`: where does a tied voter end?

Added 2026-11-17, with the same script. Everything above is about `ddo_switch`.

### Goal

How well can we predict where a voter who was tied before the debate ends (`tie`, `pro` or `con`)
from the voter alone, without the debate text? The numbers to set it against are on the same test
split, in `RESULTS.md` on branch `t2-ddo-switch-results`: always answering `tie`, a rule that uses
only the year of the debate, and the text-only ModernBERT.

### What changes for three classes

- **Metric:** macro-F1 over the three classes, and accuracy, of the predicted class. C is chosen on
  validation macro-F1. Both test numbers get a 95% interval from the bootstrap over debates (1,000
  resamples). The class weighting is the same as for `ddo_switch`.
- **Year:** the side before is `Tie` for every row, so "side before × year" is the year alone.
- **Profile:** as above.
- **Voting record:** four numbers from the voter's votes in the `ddo_undecided` training debates:
  log(1 + votes that ended `tie`), the same for `pro` and for `con`, and a flag for "no votes". A
  training row never counts its own debate. Votes in `ddo_switch` are not used.
- **Always `tie`:** the largest training class for every row. The script computes it as the floor.
- **Not computed:** AUROC and the same-debate pairs. Both need one score per voter, and a
  three-class model does not give one.

### Results

Test set: 7,091 votes from 3,649 debates, of which 6,508 end `tie`, 256 `pro` and 327 `con`.
Training set: 58,156 votes (53,181 / 2,187 / 2,788). The interval is a bootstrap over debates (1,000
resamples).

| Model | Features | Test macro-F1 | 95% interval | Test accuracy | 95% interval | F1 `tie` / `pro` / `con` |
|---|---|---|---|---|---|---|
| Always `tie` | 0 | 0.319 | 0.318–0.320 | 0.918 | 0.910–0.925 | 0.957 / 0.000 / 0.000 |
| Year | 12 | 0.382 | 0.367–0.398 | 0.788 | 0.772–0.805 | 0.887 / 0.118 / 0.141 |
| Profile | 411 | 0.342 | 0.330–0.353 | 0.584 | 0.573–0.596 | 0.741 / 0.116 / 0.168 |
| Profile, trained on unseen voters only | 411 | 0.353 | 0.340–0.365 | 0.743 | 0.731–0.754 | 0.859 / 0.105 / 0.094 |
| Voting record, earlier years only | 4 | 0.351 | 0.338–0.365 | 0.609 | 0.597–0.621 | 0.762 / 0.123 / 0.168 |
| **Voting record** | 4 | **0.423** | 0.408–0.439 | 0.748 | 0.738–0.759 | 0.867 / 0.191 / 0.213 |
| Profile + voting record | 415 | 0.407 | 0.393–0.421 | 0.732 | 0.722–0.743 | 0.858 / 0.165 / 0.197 |
| Profile + voting record + year | 427 | 0.409 | 0.395–0.424 | 0.734 | 0.724–0.745 | 0.859 / 0.167 / 0.202 |
| Year of the debate, rule (reference) | – | 0.377 | – | 0.779 | – | – |
| ModernBERT-base, text only (reference) | – | 0.360 | – | 0.644 | – | – |

The two reference rows are copied from `RESULTS.md` on branch `t2-ddo-switch-results`. The script
does not compute them, and they have no interval there. That file also has always `tie` at 0.319 and
91.8%, which the first row reproduces.

- **The voting record is the best listener-only row: macro-F1 0.423 with four features.** That is
  above the year rule (0.377), the text model (0.360) and the Year row (0.382, and the two intervals
  do not overlap).
- **No row beats always answering `tie` on accuracy** (0.918). The class weights make every model
  answer `pro` or `con` too often. The voting record model does so for 27.5% of the test rows, and
  8.2% of the voters left the tie.
- **The models find who leaves the tie, not which side they pick.** The voting record model answers
  `pro` or `con` for 491 of the 583 voters who left the tie. It names the right side for 256 of them
  (52.1%). Naming `con` for all 491 would be right for 55.8%. F1 for `pro` and `con` stays near 0.2.
- **The profile adds nothing to the voting record** (0.407 with it, 0.423 without). Adding the year
  does not help either (0.409).
- **The profile alone (0.342) and the voting record from earlier years only (0.351) are below the
  year rule (0.377).** They beat always `tie` on macro-F1 and lose to it on accuracy by more than
  0.3.

Coverage: 89.8% of test rows have a profile record, 98.4% have a voter who also votes in a training
debate, and 75.1% have a voter with training votes from an earlier year.

The counts in the second and third point come from `test_confusion` in the output JSON (rows: true
class, columns: predicted class).

### Notes

- **The rows answer `pro` or `con` at different rates,** and macro-F1 and accuracy depend on that
  rate: Year 18.0% of test rows, voting record 27.5%, profile 43.8%. The rate follows from the class
  weights and was not tuned. Part of the gaps in the table comes from this.
- **The voting record uses later votes too,** as for `ddo_switch`. With earlier years only it scores
  0.351, and 24.9% of test rows then have no record at all.
- **The comparison with the reference rows is unpaired.** Their predictions are not saved here, so
  there is no interval for the differences. `RESULTS.md` names dataset branch `t1-t2` for them and
  this script reads branch `main`. The row and class counts are the same; the rows themselves were
  not compared.
- **The Year row and the year rule use the same information.** With C = 1 or C = 10 the Year model
  scores 0.377 and 0.779, the same as the rule. Validation macro-F1 picks C = 0.001, which gives
  0.382 and 0.788. This was checked in a separate run; the script prints only the chosen C.
- **Profile, trained on unseen voters only,** scores 0.353 against 0.342 for the full profile model,
  and the intervals overlap. It has 2,876 of the 58,156 training rows and answers `pro` or `con`
  half as often (22.3% of test rows), so the two rows are hard to compare. There is no drop like
  the one for `ddo_switch`.
- **The comparisons rest on the 583 test voters who left the tie.** One run. Logistic regression
  has no random seed; the bootstrap uses seed 42.
- **The profile is a snapshot from the crawl,** taken after the debates.
- **Not in `RESULTS.md` yet,** for the same reason as the `ddo_switch` rows.
