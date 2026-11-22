# T2 and prior work: Longpre et al. (2019)

**Status:** compared on 2026-11-20. Their published numbers cannot stand next to our main T2 results:
they ask another question (which side wins a persuaded voter, not whether a voter is persuaded) on
another selection of votes. Scored on their question, our text-only `ddo_undecided` model picks the
right side for 67.1% of the tied voters who chose one, against 56.1% for the majority side; their
linguistic features reach 66.95% and all their features 69.01% in their own setting. Their model was
not retrained on our splits.

## Goal

The proposal names Longpre et al. (2019) as the prior-work baseline for DDO, "compared with their
published results where our splits match and retrained on our splits otherwise". This checks whether
the splits and the task match, and puts our models as close to their setting as the saved scores allow.

Paper: Liane Longpre, Esin Durmus and Claire Cardie. 2019. Persuasion of the Undecided: Language vs.
the Listener. Proceedings of the 6th Workshop on Argument Mining, pages 167–176.
https://aclanthology.org/W19-4519/ (read in full on 2026-11-20).

## What it does

- **A comparison of the two setups,** from their paper and our dataset (table under Results).
- **`scripts/t1_t2/which_side.py`** scores the saved `ddo_undecided` models on their question for
  tied voters. It keeps the test votes that left the tie; a model's answer is the side, Pro or Con,
  with the larger logit; the metric is accuracy, as in their paper. A constant row always answers the
  side that is more common among the training votes that left the tie. Intervals and paired tests are
  those of `scripts/t1_t2/compare_models.py`: a bootstrap over debates, and a permutation test that
  swaps whole debates. Nothing is trained.

## Run

```bash
python scripts/t1_t2/which_side.py \
    --model "Text only=outputs/ddo_undecided,outputs/ddo_undecided-seed43,outputs/ddo_undecided-seed44" \
    --model "Text + audience=outputs/ddo_undecided-audience-self,outputs/ddo_undecided-audience-self-seed43,outputs/ddo_undecided-audience-self-seed44" \
    --output outputs/which_side.json
```

It needs the `predictions_test.parquet` files of the runs in `outputs/<run>/` and streams ids and labels
from the Hub (sandbox off).

## Results

### The two setups

| | Longpre et al. (2019) | This project |
|---|---|---|
| Corpus | DDO as used by Durmus and Cardie (2018): 67,315 debates, 36,294 users | DDO as released by Durmus and Cardie (2019) |
| Question | Which side, Pro or Con, does a persuaded voter end on? | Does a decided voter switch (`ddo_switch`)? Does a tied voter end on tie, Pro or Con (`ddo_undecided`)? |
| Tied voters | Only those who picked a side: 4,360 votes from 3,652 debates | All of them: 72,410 votes, of which 6,155 picked a side |
| Decided voters | Only those who switched: 2,642 votes from 2,183 debates | All of them: 60,409 votes, of which 3,867 switched |
| Given to the model | Features of the debate and of the audience member; not the voter's side before the debate | The debate text, the voter's side before the debate and, for text + audience, the voter's stance profile |
| Audience features | Mostly the voter against each debater: same political or religious ideology, similarity of their big-issue answers, same gender. Also how many issues the voter has a view on, and "persuadability": the share of the voter's other training debates in which they changed their stance | The voter's own six stance labels; nothing about the debaters. The voting record of our extension rows is the counterpart of "persuadability" |
| Model | Logistic regression on hand-built linguistic and audience features | Fine-tuned ModernBERT-base; logistic regression for the baselines without text |
| Protocol | 5-fold cross-validation; whether a debate's votes stay in one fold is not stated | One split by debate, 80/10/10 |
| Metric | Accuracy; the majority side has 57.43% (tied) and 59.42% (decided) | AUROC and F1 of the switch class; macro-F1 for tied voters |

- **The splits do not match, and neither does the task.** They predict the direction of persuasion
  among voters who were persuaded. We predict whether persuasion happens, among all voters. Their
  data leave out the 94% of decided voters who stay and the 91.5% of tied voters who stay tied, which
  are the larger classes of our two tasks.
- **Their decided-voter task has no counterpart in our models.** Our models are given the voter's side
  before the debate, and a voter who switches ends on the other side, so for a switcher the answer to
  "which side" follows from the input.

### Tied voters, scored on their question

583 of our 7,091 `ddo_undecided` test votes left the tie (256 Pro, 327 Con), from 472 debates; 4,975
training votes did (2,187 Pro, 2,788 Con). Accuracy of the side with the larger logit; a model row is
the average over seeds 42, 43 and 44, with the 95% interval of the average.

| Row | Accuracy | 95% interval | Per seed | Setting |
|---|---|---|---|---|
| Always Con | 56.1% | 51.4–60.3 | | ours, 583 test votes |
| Text only | 67.1% ± 0.4 | 63.5–70.6 | 66.7, 67.4, 67.2 | ours |
| Text + audience | 63.9% ± 1.8 | 60.3–67.5 | 65.9, 62.4, 63.3 | ours |
| Majority side | 57.43% | | | theirs: Table 3, 5-fold cross-validation, 4,360 votes |
| Audience features | 61.47% | | | theirs |
| Linguistic features | 66.95% | | | theirs |
| All features | 69.01% | | | theirs |

| Comparison (first minus second) | Accuracy difference | 95% interval | p |
|---|---|---|---|
| Text only − Always Con | +11.0 points | +6.1 to +16.5 | 0.0005 |
| Text + audience − Always Con | +7.8 points | +2.8 to +13.2 | 0.0030 |
| Text + audience − Text only | −3.3 points | −5.5 to −1.1 | 0.0035 |

- **The debate text tells which side wins a tied voter.** The text-only model is 11 points above the
  majority side, in the range their linguistic features reach in their setting (66.95%).
- **So the weak `ddo_undecided` scores come from the other half of the task:** telling who leaves the
  tie at all. On the three-class task both models have a macro-F1 near 0.38 with 91.8% of the test
  votes tied (`analysis/2026-11-19_t2-undecided-audience/`).
- **The profile does not help here; it costs 3.3 points.** Their paper finds the opposite for tied
  voters (66.95% to 69.01% when audience features are added). The two are not the same test: their
  audience features compare the voter with each debater, and our profile line describes the voter
  only.
- **Their "persuadability" feature is our voting record.** In their task, removing it changes almost
  nothing (61.47% to 61.46% for tied voters). In ours it is the strongest signal without text (AUROC
  0.790 on `ddo_switch`, an extension of the proposal). That fits: how often a voter changed before
  tells whether they change, not towards which side.

## Notes

- **The two columns of numbers are not one experiment.** Other corpus version, other selection of
  votes, cross-validation against one held-out split, and their folds may share debates between
  training and test. The numbers show the same range and no more.
- **Our models were not trained for this question.** They were trained on all tied voters with the tie
  as a third class and class weights; the which-side answer reads two of their three logits. A model
  trained on the persuaded voters only might score differently. Not tested.
- **Not done: their model retrained on our splits,** the proposal's fallback. It needs their feature
  set (lexicons for subjectivity, connotation, politeness and argumentation, and the debaters'
  profiles). A reduced version, logistic regression on TF-IDF of the debate text, would be a
  text baseline without a GPU; it is not built.
- **583 test votes** give intervals of about ±3.5 points.
- **Checked:** the script reads the same rows and labels as the comparison script (it uses its
  functions); the constant row equals the share of Con among the 583 votes (327 / 583 = 56.1%); a
  direct calculation from two of the prediction files gives the script's accuracy for those seeds.
