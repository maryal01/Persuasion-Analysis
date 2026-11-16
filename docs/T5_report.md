# T5: Predicting Whether a CMV Author Awards a Delta

## 1. Task and Dataset

**Task.** Given a post on Reddit's r/changemyview (CMV), predict whether its author (the original poster, or OP) will award at least one delta (∆) to a commenter. A delta indicates that a commenter successfully changed the OP's view.

**Dataset.** We use Webis-CMV-20 (Al-Khatib et al., 2020), covering 2013–2017. We select posts whose authors have available LIWC and Receptiviti features, resulting in **8,121 posts from 4,641 unique authors**.

We split the dataset by author, ensuring that posts from the same OP never appear in different splits.

| Split      | Posts | Delta rate |
| ---------- | ----: | ---------: |
| Train      | 6,626 |        63% |
| Validation |   725 |        67% |
| Test       |   770 |        67% |

**Labels.** We assign a positive label when DeltaBot confirms that the OP awarded a delta. We do not use the dataset's original `delta` field because it is derived from post flairs, which are missing before March 2015 and sometimes replaced by other flairs. In our dataset, the flair-based label misses **2,192 of the 5,179 DeltaBot-confirmed positive posts**.

Some label uncertainty remains. In 331 posts (4%), the OP attempted to award a delta, but DeltaBot did not confirm it. Of these, 132 were explicitly rejected, often because the explanation was too short, while 199 had no bot response. These posts are labeled negative.

**Text preprocessing.** We remove moderator footnotes, paragraphs beginning with post-hoc `Edit:` or `Update:` markers, and explicit delta symbols. These can reveal information unavailable when the original post was submitted. Cleaning removes 20.8% of the characters, mostly repetitive moderator footnotes. A small number of unusual edit formats remain (71 lines across 65 posts).

## 2. Features and Baseline Models

We evaluate logistic regression models using different inputs:

- **Post text (Bag of Words):** TF-IDF features representing the words in the current post.
- **Post length:** Length of the cleaned post.
- **Posting date:** When the post was submitted, included to check for temporal effects.
- **LIWC (93 features):** Linguistic characteristics derived from an author's writing, such as emotion words, pronouns, and word usage patterns.
- **Receptiviti (118 features):** Psycholinguistic characteristics inferred from writing, including personality-related measures.
- **Combined features:** Post text together with all 211 LIWC and Receptiviti features.

Logistic regression learns a weight for each input feature and produces a probability of awarding a delta.

We select the regularization type (L1 or L2) and strength (C = 0.001, 0.01, 0.1, 1, 10) using validation AUROC. The selected models are evaluated on the held-out test set.

### Logistic Regression Results

| Model                       |     AUROC | 95% CI      |    PR-AUC | Balanced Acc. |  Macro-F1 |
| --------------------------- | --------: | ----------- | --------: | ------------: | --------: |
| Always predict delta        |     0.500 | N/A         |     0.668 |         0.500 |     0.400 |
| Posting date only           |     0.589 | 0.543–0.634 |     0.731 |         0.553 |     0.540 |
| Post length only            |     0.500 | N/A         |     0.668 |         0.500 |     0.400 |
| Post text (TF-IDF)          |     0.564 | 0.520–0.610 |     0.720 |         0.541 |     0.537 |
| LIWC                        |     0.705 | 0.660–0.748 |     0.829 |         0.660 |     0.605 |
| Receptiviti                 | **0.726** | 0.681–0.769 | **0.841** |     **0.677** | **0.647** |
| LIWC + Receptiviti          |     0.718 | 0.672–0.766 |     0.835 |         0.667 |     0.611 |
| TF-IDF + LIWC + Receptiviti |     0.718 | 0.672–0.766 |     0.835 |         0.667 |     0.611 |

**Evaluation metrics.** AUROC is our primary metric. It measures how well a model ranks positive examples above negative ones, with 0.5 representing random ranking and 1.0 perfect ranking. PR-AUC focuses on positive-class precision and recall; its no-skill baseline is 0.668 because 66.8% of test posts are positive. Balanced accuracy averages recall across the two classes, while macro-F1 averages their F1 scores, preventing the majority class from dominating the evaluation.

The 95% confidence intervals are obtained through **author-level bootstrap resampling**: authors are sampled with replacement, keeping their associated posts together. This accounts for the fact that multiple posts from the same author are not independent.

### Baseline Findings

- **Author features are substantially more predictive than post text.** LIWC and Receptiviti achieve AUROC values of 0.705 and 0.726, compared with 0.564 for TF-IDF.
- **Adding post words does not improve the combined logistic regression model.** The selected L1-regularized model assigns zero weight to all TF-IDF features, producing the same predictions as LIWC + Receptiviti alone.
- **Post length provides no useful predictive performance** under the selected configuration.
- **Posting date contains some predictive signal** (AUROC 0.589). The positive-label rate increases from 53% in 2013 to 76% in 2017.
- Receptiviti has the highest observed baseline AUROC, but the differences among LIWC, Receptiviti, and their combination are small relative to test uncertainty. Establishing whether these differences are statistically significant would require paired comparisons.

For context, Al-Khatib et al. (2020) report AUROC values of 0.52 for word features and 0.62 for personality traits. However, their labels and evaluation setup differ from ours, so the numbers are not directly comparable.

## 3. ModernBERT Experiments

We fine-tune `answerdotai/ModernBERT-base` using two configurations.

**Model A: Post text only.** The cleaned post is encoded by ModernBERT. We apply masked mean pooling to its token representations and use a linear classification layer to predict whether the OP awards a delta.

**Model B: Post text + author features.** ModernBERT encodes the current post. In parallel, a small neural network transforms the 211 LIWC/Receptiviti features into a 64-dimensional representation. We concatenate the text and author representations before classification.

### Training Configuration

| Setting                 | Value                              |
| ----------------------- | ---------------------------------- |
| Model                   | ModernBERT-base                    |
| Maximum sequence length | 2,048 tokens                       |
| Epochs                  | 3                                  |
| Effective batch size    | 32 (8 × 4 gradient accumulation)   |
| Optimizer               | AdamW                              |
| Encoder learning rate   | 2e-5                               |
| New-layer learning rate | 1e-3                               |
| Precision               | bfloat16                           |
| Model selection         | Best validation AUROC              |
| Random seeds            | 0, 1, 2                            |
| GPU memory              | Approximately 13.5 GiB             |
| Training time           | Approximately 7–10 minutes per run |

The newly initialized layers use a higher learning rate than the pretrained encoder. We compared two head learning rates on validation for the author-feature variant: 1e-3 achieved AUROC 0.716, versus 0.666 for 2e-5. We retained 1e-3. No other hyperparameters were systematically tuned.

Each ModernBERT configuration was trained with three random seeds. Results report the mean and standard deviation across these runs.

### ModernBERT Results

| Model                                   |             AUROC |        PR-AUC | Balanced Acc. |      Macro-F1 |
| --------------------------------------- | ----------------: | ------------: | ------------: | ------------: |
| Logistic Regression: TF-IDF             |             0.564 |         0.720 |         0.541 |         0.537 |
| **ModernBERT: post only**               | **0.593 ± 0.010** | 0.737 ± 0.007 | 0.565 ± 0.018 | 0.519 ± 0.046 |
| Logistic Regression: LIWC + Receptiviti |             0.718 |         0.835 |         0.667 |         0.611 |
| Logistic Regression: Receptiviti only   |         **0.726** |     **0.841** |     **0.677** |     **0.647** |
| **ModernBERT: post + author features**  | **0.706 ± 0.002** | 0.834 ± 0.004 | 0.650 ± 0.010 | 0.626 ± 0.009 |

The ± values represent variation across training seeds, not 95% confidence intervals.

### ModernBERT Findings

**1. ModernBERT modestly improves text-only prediction.**

ModernBERT achieves AUROC 0.593, compared with 0.564 for TF-IDF. The improvement is small, and post-text-only performance remains relatively weak.

**2. Adding ModernBERT does not improve author-feature prediction.**

ModernBERT with author features achieves AUROC 0.706, below logistic regression using LIWC + Receptiviti (0.718) or Receptiviti alone (0.726).

This suggests that, under our tested setup, most of the useful predictive information comes from the author features rather than the current post's text.

**3. Results are consistent across random seeds.**

The individual test AUROCs were:

- Post only: 0.584, 0.604, 0.591.
- Post + author features: 0.708, 0.706, 0.704.

Standard deviations are small (0.010 and 0.002). The author-feature model is particularly stable across seeds.

**4. The models show similar behavior on validation.**

ModernBERT with author features achieves validation AUROC 0.709 ± 0.006, compared with approximately 0.72 for the logistic regression baseline.

The best checkpoint generally occurs at epoch 2 or 3, suggesting that three epochs are sufficient for the tested configuration.

## 4. Limitations

**Potential temporal leakage in author features.** LIWC and Receptiviti are precomputed once per author from their writing history. The released data does not establish whether all source texts were written before each target post. Some features may therefore incorporate future information, potentially making the reported author-feature results optimistic.

**Limited test size.** The test set contains 770 posts. The individual model AUROC confidence intervals have a half-width of approximately 0.045. Consequently, small differences should not be overinterpreted. Low variation across training seeds does not eliminate this evaluation uncertainty.

**Limited hyperparameter tuning.** We tested one main ModernBERT training configuration, with a small comparison of new-layer learning rates. Alternative pooling methods, architectures, or optimization settings might improve results.

**Text preprocessing.** Without cleaning, TF-IDF AUROC increases from 0.564 to 0.603, and the post-length baseline increases from 0.500 to 0.560. This indicates that moderator footnotes and post-hoc edits introduce predictive signals that are not necessarily available at submission time.

## 5. Conclusion

Our experiments show that **LIWC and Receptiviti author features provide substantially stronger predictive signals for delta-awarding behavior than the current post text**.

ModernBERT improves text-only prediction modestly but does not outperform simple logistic regression when author features are available. These findings suggest that information about the author may be more useful than the content of the current post for this task.

However, the temporal validity of the precomputed author features remains an important limitation. The results establish predictive performance under the released dataset setup, but do not yet demonstrate equally strong prediction using exclusively information available before each post was submitted.
