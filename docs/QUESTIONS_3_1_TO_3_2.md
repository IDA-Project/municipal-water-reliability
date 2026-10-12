> Historical section retained for traceability. The current full-project answers and consistent experiment are in [QA_CORRECTED_ANSWERS.md](QA_CORRECTED_ANSWERS.md) and [QA_REVIEW.md](QA_REVIEW.md). Do not combine older metrics or scope statements with the corrected QA results.

# Questions 3.1 and 3.2: model training

This answer continues the team's WSA-level binary classification design from Questions 1–2.4. The target is `low_reliability`, where `1` means that less than 60% of the WSA population receives reliable water supply and `0` means that at least 60% receives reliable supply.

## 3.1 Naïve baseline

We use a majority-class `DummyClassifier` as a transparent naïve baseline. It ignores every predictor and always predicts the most common class in each training fold. On the fixed development set, class `0` is the majority class, with 84 of 115 observations, while class `1` contains 31 observations.

This baseline establishes the minimum reference that a useful classifier must improve upon. It also shows why accuracy alone is unsuitable for this imbalanced problem: a model can predict only the majority class and still appear reasonably accurate while failing to identify any low-reliability WSA. We therefore use macro-F1 as the primary comparison measure because it gives both classes equal importance. Across the repeated development folds, the baseline produces a mean macro-F1 of **0.422 ± 0.006**. The reserved test set is not used.

## 3.2 Four substantive models

We train four substantive classifiers on the same fixed development partition and the same 25 repeated-stratified cross-validation folds (5 folds repeated 5 times, random seed 42). Preprocessing and random oversampling occur inside each training fold, preventing information from the corresponding validation fold from leaking into model fitting. All models use the same leakage-safe features established in Question 2.4, and none uses the WSA name or a reliable-supply value from which the target is derived.

| Model | Category | Role in the comparison |
| --- | --- | --- |
| Logistic Regression | Traditional | Provides a regularised, transparent linear decision boundary. |
| Random Forest | Traditional | Captures non-linear relationships and interactions between population and access features. |
| Multilayer Perceptron (MLP) | Neural | Uses two dense hidden layers, ReLU activation, Adam optimisation and early stopping. |
| Residual Tabular Neural Network | Neural | Uses an input projection and a two-layer residual block designed for the engineered tabular features. |

The majority baseline is not counted among these four substantive models. The residual tabular network is the proposed task-appropriate second neural architecture and requires the lecturer's approval, as specified by the template. Its selection, parameters and approval status are recorded in `reports/q3/model_training_manifest.json`.

As a training sanity check, the mean development validation macro-F1 values are 0.800 for Logistic Regression, 0.793 for Random Forest, 0.786 for the MLP and 0.787 for the Residual Tabular Neural Network. Each substantive model therefore improves substantially on the naïve baseline under the same validation protocol. These values are reproducible in `reports/q3/development_cv_training_summary.csv` and are not held-out test results.

Each fitted pipeline is saved locally under `models/q3/`. The development cross-validation values are training evidence only, not final test performance. The 29-row test partition remains reserved for the later evaluation question and no test predictions or test metrics are produced in Questions 3.1–3.2.

## Assumptions and scope boundary

- The fixed Question 2.3 split assignments remain authoritative.
- The four models use fixed, reasonable initial settings; Question 3.2 requires training rather than hyperparameter optimisation.
- The residual tabular neural network remains provisional until lecturer approval is documented.
- Questions 3.3 onward, including tuning, full comparison metrics and final held-out testing, remain outside this implementation.
