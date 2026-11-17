# Results

Test-set results for each trained model. Each row is one training run; add new rows as runs finish.

## T1: CMV persuasion (`cmv`)

Test set: 1,019 delta/no-delta pairs (2,038 replies) from posts that never appear in training
(dataset branch `t1-t2`). The main metric is **pairwise accuracy**: how often the model scores the
delta reply above its no-delta partner. Chance is 50%.

| Model | Pairwise acc. | AUROC | Accuracy | F1 (delta) | Macro-F1 |
|---|---|---|---|---|---|
| Coin flip | 50.0% | 0.500 | – | – | – |
| More comments wins | 53.7% | – | – | – | – |
| More links wins | 54.3% | – | – | – | – |
| Longer reply wins | 55.3% | – | – | – | – |
| **ModernBERT-base, text only** | **66.1%** | **0.629** | 55.9% | 0.645 | 0.532 |

- Model: [`nlp-final-project11/nlp-project-modernbert-cmv`](https://huggingface.co/nlp-final-project11/nlp-project-modernbert-cmv)
  (private). Trained on Gilbreth (one A100, ~1h45m, Slurm job 12026112), 3 epochs, max 4,096 tokens.
- With 1,019 pairs, pairwise accuracy has a rough 95% interval of ±3 points (about 63–69%).
- Single-reply accuracy is low because the model labels 74% of replies "delta" (precision 0.54,
  recall 0.80). Its ranking is better than its yes/no cutoff, which is why the pairwise score is higher.
- No delta markers ("Δ", "!delta") leak into the replies: they appear in under 0.5% of replies on either side.

**Caveats**
- One run, one random seed. Final numbers should average a few seeds.
- This run picked its checkpoint by validation F1 (step 1000, validation AUROC 0.630). Later
  checkpoints had higher validation AUROC (0.644). The code now selects by AUROC.
- Text only. Audience features (RQ2) haven't been added yet.

**Prior work.** Al-Khatib et al. (ACL 2020), who built Webis-CMV-20, report per-comment accuracy
rather than pairwise: word count 51.9%, Tan et al.'s best text features 57.8%, their model with
debater traits 61.1%. Their metric differs from ours, so don't compare the numbers directly.
Tan et al. (2016)'s 65.1% is on a different corpus.

## T2: DDO voter switching (`ddo_switch`, `ddo_undecided`)

Not trained yet.

## T5: Predicting Whether a CMV Author Awards a Delta

### Task and Dataset

We predict whether the original poster (OP) of a Reddit Change My View (CMV) post awards at least one delta (∆), indicating that a commenter changed their view.

We use **8,121 posts from 4,641 authors** in Webis-CMV-20 (2013–2017), selecting authors with available LIWC and Receptiviti features. The dataset is split by author to prevent the same OP from appearing in multiple splits.

| Split      | Posts | Delta rate |
| ---------- | ----: | ---------: |
| Train      | 6,626 |        63% |
| Validation |   725 |        67% |
| Test       |   770 |        67% |

**Labels:** Positive labels are based on DeltaBot-confirmed delta awards rather than the dataset's unreliable flair-based labels, which miss 2,192 confirmed positive posts.

**Preprocessing:** We remove moderator footnotes, post-hoc `Edit/Update` paragraphs, and delta symbols to reduce information leakage.

### Models

We evaluate logistic regression baselines using post text (TF-IDF), posting date, post length, LIWC (93 linguistic features), Receptiviti (118 psycholinguistic features), and their combinations. Hyperparameters (L1/L2 regularization and C) are selected using validation AUROC.

We also fine-tune **ModernBERT-base** in two configurations:

- **Post only:** ModernBERT encodes the current post, followed by mean pooling and classification.
- **Post + author features:** ModernBERT text embeddings are concatenated with a learned 64-dimensional representation of the 211 LIWC/Receptiviti features.

ModernBERT is trained for 3 epochs with an effective batch size of 32, learning rates of 2e-5 (encoder) and 1e-3 (new layers), and three random seeds. The best checkpoint is selected using validation AUROC.

### Results

All results are evaluated on the same 770-post test set. **AUROC is the primary metric** (0.5 = random ranking, 1.0 = perfect ranking). PR-AUC evaluates positive-class precision and recall, while balanced accuracy and macro-F1 account for class imbalance.

ModernBERT results are reported as **mean ± standard deviation across three seeds**. The logistic regression results come from a single fitted model per configuration.

| Model                                            |         AUROC |        PR-AUC | Balanced Acc. |      Macro-F1 |
| ------------------------------------------------ | ------------: | ------------: | ------------: | ------------: |
| Always predict delta                             |         0.500 |         0.668 |         0.500 |         0.400 |
| Posting date only                                |         0.589 |         0.731 |         0.553 |         0.540 |
| Post length only                                 |         0.500 |         0.668 |         0.500 |         0.400 |
| Logistic Regression: post text (TF-IDF)          |         0.564 |         0.720 |         0.541 |         0.537 |
| Logistic Regression: LIWC                        |         0.705 |         0.829 |         0.660 |         0.605 |
| Logistic Regression: Receptiviti                 |     **0.726** |     **0.841** |     **0.677** |     **0.647** |
| Logistic Regression: LIWC + Receptiviti          |         0.718 |         0.835 |         0.667 |         0.611 |
| Logistic Regression: TF-IDF + LIWC + Receptiviti |         0.718 |         0.835 |         0.667 |         0.611 |
| ModernBERT: post only                            | 0.593 ± 0.010 | 0.737 ± 0.007 | 0.565 ± 0.018 | 0.519 ± 0.046 |
| ModernBERT: post + LIWC + Receptiviti            | 0.706 ± 0.002 | 0.834 ± 0.004 | 0.650 ± 0.010 | 0.626 ± 0.009 |

The test-set positive rate is 66.8%, giving a no-skill PR-AUC of 0.668. Logistic regression AUROC confidence intervals were estimated using author-level bootstrap resampling and are provided in the [detailed T5 report](docs/T5_report.md).

### Key Findings

1. **Author features outperform post text.** LIWC and Receptiviti achieve 0.705–0.726 AUROC, compared with 0.564 for TF-IDF and 0.593 for text-only ModernBERT.

2. **ModernBERT provides limited improvement.** It slightly improves text-only prediction but does not outperform logistic regression when author features are available (0.706 vs. 0.718–0.726 AUROC).

3. **Adding post text does not improve the combined logistic regression baseline.** The validation-selected L1 model assigns zero weight to all TF-IDF features, retaining only 20 of the 211 author features. This suggests that the text provides limited additional information under the tested linear configuration.

4. **Posting date has some predictive signal** (0.589 AUROC), while post length offers no useful improvement.

5. **Small performance differences remain uncertain.** The test set contains 770 posts, and individual AUROC confidence intervals have an approximate half-width of 0.045.

### Limitations

- **Potential temporal leakage:** LIWC/Receptiviti features may incorporate writing produced after the target post, potentially inflating author-feature performance.
- **Limited test size:** Small AUROC differences should not be overinterpreted.
- **Limited tuning:** ModernBERT was evaluated with one main training configuration, apart from a comparison of new-layer learning rates.
- **Feature scaling:** Differences in feature scales and regularization may affect the contribution of TF-IDF features to the combined logistic regression model.

**Conclusion:** Author-level linguistic and psycholinguistic features are stronger predictors of delta-awarding behavior than current post text in our experiments. ModernBERT does not provide a clear advantage over simpler models when these features are available.

For complete confidence intervals, preprocessing details, architecture, training configurations, and analysis, see the [detailed T5 report](docs/T5_report.md).
