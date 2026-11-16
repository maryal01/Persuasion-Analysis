# T2 listener-only baselines for `ddo_switch`

**Status:** done for `ddo_switch` (2026-11-16). Not run for `ddo_undecided`, which has three classes.

## Goal

How well can we predict whether a decided voter switches sides from the voter alone, without the
debate text? This is the "listener" side of RQ2. The number to set it against is the text-only
ModernBERT on the same test rows (AUROC 0.719).

## What it does

`scripts/t1_t2/listener_baseline.py` fits class-weighted logistic regressions on the rows and splits
of the Hub dataset `ddo_switch` (branch `t1-t2`). It reads only the id and label columns, so the
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
python scripts/t1_t2/listener_baseline.py --users ddo/users.json --output outputs/listener_baseline.json
```

Needs scikit-learn (added to `requirements.txt`) and `users.json` from the DDO corpus. It takes about
a minute on a laptop.

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
