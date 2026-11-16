# T3 baselines: stance from a user's text

**Status:** done for the majority, random and TF-IDF baselines (2026-11-16). ModernBERT is not run yet.

## Goal

How well can a simple model predict a DDO user's self-reported stance from what they wrote? These are
the reference rows for the T3 ModernBERT model, on the dataset from PR #27 (`ddo_users` and
`ddo_history`, Hub branch `t3`).

## What it does

`scripts/t3/baselines.py` scores four models on held-out users, for each of the six label columns of
`ddo_users`:

- **Majority:** always the most common class among training users. This is the bar for accuracy.
- **Random:** a class drawn at random with the training class frequencies, averaged over 1,000
  draws. This is the bar for macro-F1: a class-weighted model with no signal scores like it, not like
  the majority row.
- **TF-IDF, class-weighted:** logistic regression on TF-IDF word features, classes weighted equally,
  C chosen on validation macro-F1.
- **TF-IDF, unweighted:** the same without class weights, C chosen on validation accuracy.

A user's text is every `ddo_history` item they wrote (debate arguments, comments, opinion arguments,
poll votes), each with its title in front. Votes have no text and are not used. Items from debates in
T2's validation or test split are left out (`debate_split`), so a model built this way can later write
profiles for T2 without having read those debates.

Class names, labels and splits are read from the dataset. Only users with the label and at least one
text item are scored. The vocabulary comes from training users (68,840 words used by at least 5 of
them), and the test split is scored once per model.

## Run

```bash
python scripts/t3/baselines.py --smoke                                # about 30 seconds
python scripts/t3/baselines.py --output outputs/t3_baselines.json     # 4 to 12 minutes on a laptop
```

Needs scikit-learn (added to `requirements.txt`) and read access to the dataset repo. Both tables are
streamed from the Hub.

## Results

18,397 labelled users have usable text (median 1,063 words; a tenth have under 97). Intervals are 95%,
from a bootstrap over test users (1,000 resamples). The accuracy difference is computed on the same
resamples for both models.

Accuracy, against the majority class:

| Label | Test users | Majority | TF-IDF, unweighted | Difference |
|---|---|---|---|---|
| economy | 1,164 | 0.497 | 0.523 | +0.027 (+0.005 to +0.048) |
| social_issues | 1,427 | 0.434 | 0.505 | +0.071 (+0.043 to +0.097) |
| personal_liberties | 1,271 | 0.505 | 0.517 | +0.012 (+0.004 to +0.020) |
| foreign_policy | 991 | 0.571 | 0.582 | +0.011 (−0.004 to +0.027) |
| religion | 1,374 | 0.468 | 0.593 | +0.125 (+0.098 to +0.154) |
| political_ideology_group | 1,122 | 0.325 | 0.374 | +0.049 (+0.019 to +0.078) |

Macro-F1, against random guessing:

| Label | Classes | Random | TF-IDF, class-weighted | Its accuracy |
|---|---|---|---|---|
| economy | 3 | 0.333 | 0.418 (0.391–0.447) | 0.448 |
| social_issues | 3 | 0.333 | 0.462 (0.439–0.488) | 0.483 |
| personal_liberties | 3 | 0.333 | 0.450 (0.420–0.480) | 0.512 |
| foreign_policy | 3 | 0.333 | 0.424 (0.392–0.455) | 0.484 |
| religion | 4 | 0.250 | 0.395 (0.370–0.421) | 0.539 |
| political_ideology_group | 5 | 0.200 | 0.345 (0.316–0.376) | 0.359 |

- **Text carries a real but small signal.** Accuracy is 1 to 12.5 points above the majority class, and
  macro-F1 is 0.09 to 0.15 above random guessing.
- **Religion and social issues are the most predictable.** Foreign policy is not clearly above the
  majority class, and personal liberties and economy are barely above it.
- **The two TF-IDF models trade the two metrics.** The class-weighted one has the better macro-F1 but
  falls below the majority class in accuracy on economy (−0.048) and foreign policy (−0.087). The
  unweighted one keeps accuracy but mostly predicts the large classes: its macro-F1 on personal
  liberties (0.256) and foreign policy (0.317) is below random guessing.
- **"mixed" is the hardest class.** The class-weighted model finds 28% to 44% of the mixed users. When
  a user is at one of the two poles and the model picks a pole, it picks the right one in 66% (foreign
  policy) to 77% (personal liberties) of cases.

The full output, with all four models, both metrics and the confusion matrices, is written by
`--output`.

## Notes

- **More text helps, except for economy.** With the class-weighted model, users with 10,000 or more
  words are classified correctly in 62.7% of cases for religion (majority 36.0% in that group), 58.1%
  for social issues (45.3%) and 48.4% for the ideology group (35.3%). Users with under 100 words are
  below the majority class on every label. Economy stays between 37% and 48% at every length.
- **Checked three ways,** with one-off scripts that are not in the repo. A second implementation
  (pyarrow reader, pandas, scikit-learn metrics) reproduces the TF-IDF numbers. With shuffled training
  labels (six shuffles per label, C fixed at 1) macro-F1 averages 0.33 on the three-class labels, 0.24
  on religion and 0.20 on the ideology group, close to the random row (0.33, 0.25, 0.20). The heaviest
  words per class are sensible, for example "christ" and "sin" for christian, "quran" and "allah" for
  other religion, "atheist" and "evidence" for nonreligious.
- **Titles and held-out debates make little difference.** Without titles every number moves by at
  most 0.013. With `--all-debates` macro-F1 moves by −0.02 to +0.03 and 726 more users get text. So
  leaving out T2's validation and test debates costs nothing measurable.
- **About 10% of labelled users have no usable text** (for example 1,207 of 13,070 for economy). They
  are not scored. A text model cannot profile them, or the T2 voters who only voted.
- **Votes are not used.** Vote rows dated before about September 2008 are stored as "tied before and
  after", although that form most likely never asked (see PR #24). PR #28 empties their side fields
  in `ddo_history`; votes can be used as a feature once the table is rebuilt.
- **One run, no seeds.** Logistic regression is deterministic here; the intervals cover the choice of
  test users only.
- **Not in `RESULTS.md` or the `AGENTS.md` status table yet.** Both are changed by open PRs (#22,
  #25); the rows go in once those merge.
