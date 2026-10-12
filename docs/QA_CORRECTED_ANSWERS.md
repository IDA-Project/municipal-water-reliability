# Corrected report text and evidence guide

> Follow-up code recheck: the app has now been integrated with the corrected packaged pipeline. Three valid and one invalid UI cases pass, as does a clean installation without raw data or training files. See [DEPLOYMENT_READINESS.md](DEPLOYMENT_READINESS.md). Earlier statements below about the app being unchanged describe the original QA scope; no hosted deployment has occurred.

Team-review draft based on the consistent 6 October 2026 QA experiment. Add named evidence to the final PDF, actual citations, approvals and signed contributions. Do not combine these results with historical alternate-pipeline notebook numbers. The original holdout has already been disclosed; this is a QA reproduction check.

## Q1: problem, task and governance

### 1.1

We investigate unequal reliability of water supply in South Africa. Households, schools, clinics and businesses depend on this service. Municipal water-service planners and DWS are the intended stakeholders. The proposed benefit is prioritising investigation, not predicting pipe failures or automatically allocating resources.

One record represents a WSA, not a ward or household. Inputs are aggregate population, access-to-water and urban/rural composition measures. Class 1 means reported reliable-supply percentage below 60%; class 0 means at least 60%. This team threshold needs stakeholder justification and is not an official DWS category. The task is cross-sectional classification, not next-month forecasting. The export filename does not establish the underlying measurement period. When observed reliability is available, apply the transparent threshold directly rather than approximate its label with ML.

### 1.2

Macro-F1 is primary, supported by low-reliability precision/recall, accuracy, weighted-F1, ROC-AUC and average precision (AP). A proposed operational indicator is the fraction of independently verified high-need WSAs captured under a fixed investigation budget; the stakeholder must agree the budget and verification process. Actual improved water delivery has not been measured. False positives divert scarce investigation capacity; false negatives leave high-need WSAs overlooked. Class 0 is not proof of uninterrupted service for every resident.

### 1.3

Actual source: DWS NIWIS Water Supply Reliability exports. No asset-register dataset is supplied and CC BY is not established. The source URL, recorded terms and hashes are in `docs/DATA_SOURCES.md`. Current network access could not independently recheck the publisher terms. Redistribution permission, written lecturer source approval, the measurement period and reliability denominator remain to be confirmed.

The extract contains 144 aggregate WSAs and overall/urban/rural population, access and reliable-supply measures. It has no billing records, pipe ages, maintenance logs or monthly event histories. Aggregation reduces privacy risk but does not certify POPIA compliance. Avoid personal-data joins, detailed infrastructure disclosures and automated consequential decisions. Report source anomalies, subgroup supports and uncertainty. Human review and current official information are required. No demographic fairness guarantee is made.

## Q2: audit, EDA and preparation

### 2.1

Insert `reports/q2/data_dictionary.csv` and the audit tables. Verified: 144 unique WSAs, no duplicate identifiers, 105 class-0 and 39 class-1 records, 45 missing cells. There are 57 component-above-population flags affecting 22 WSAs; retain and investigate them rather than silently changing source values. IQR flags are review indicators, not automatic deletion rules. Eight of nine national controls reconcile; overall access differs by 552,511 alongside nine missing WSA access totals. The national categories are not training observations. Reliable-supply counts/percentages directly define or reconstruct the label and are excluded from predictors.

### 2.2

Insert all five figures from `reports/q2/figures/`, with `reports/q2/interpretations.md`: binary class distribution, missingness, access versus reliability, rural share versus reliability, and urban/rural comparison. Include units, readable labels and caveats about source anomalies, ecological inference and correlation rather than causation. A continuous reliability histogram alone is not the binary class-balance evidence.

### 2.3

Saved seed-42 stratified assignments contain 115 development WSAs (84/31) and 29 test WSAs (21/8). Model selection uses five stratified development folds repeated five times. Fit transformations only on each training fold; validation/test retain natural class proportions. Per-record dates are absent, so a temporal split is unavailable. Unique WSA IDs are disjoint across partitions. Province inference from export row order is unverified; existing grouped summaries show class counts, not measured province generalisation. Bootstrap uncertainty is now supplied; no threshold-sensitivity experiment has been completed. Disclose original test exposure.

### 2.4

Deterministic cleaning normalises headers and parses counts/percentages without learning statistics. Missing/out-of-range target percentages are excluded from labelled modelling, not assigned class 0. The complete fitted pipeline selects six safe raw predictors, derives access rates and population shares plus a dominant-composition category, median-imputes numeric variables with missingness indicators, mode-imputes/one-hot encodes the category, robust-scales numeric values, oversamples training cases only, and fits the estimator. Undefined ratios stay unavailable until imputation. Reliability columns and WSA names cannot enter the engineered predictor matrix.

## Q3: baseline and model set

### 3.1

The majority DummyClassifier ignores features and predicts class 0. Development macro-F1 is 0.422 ± 0.006; test accuracy is 0.724 and positive recall is zero. This shows why accuracy alone can conceal failure to identify low-reliability WSAs. It is not counted among four substantive models.

### 3.2–3.3

Use Logistic Regression, Random Forest, MLP and a residual tabular network throughout. They share saved partitions, all 25 development folds, preprocessing and imbalance treatment. A deeper MLP is not independently established as a distinct approved architecture. The residual network has an input projection, two-layer residual block and sigmoid output. Tabular, nonsequential data do not justify imagery or sequence architectures. Lecturer approval for the residual architecture remains unrecorded.

### 3.4

| Model | Configuration |
| --- | --- |
| Logistic | C=1, L2 (`l1_ratio=0`), liblinear, max_iter=2000, no class weighting; logistic cross-entropy with L2 regularisation. Common training-only oversampling. |
| Random Forest | 300 trees, min_samples_leaf=2, max_features=sqrt, default Gini splits, unlimited depth, one estimator worker, seed 42. Neural loss/epochs/batch size do not apply. |
| MLP | ReLU hidden layers 32/16, sigmoid binary output, Adam at 0.001, batch=16, alpha=0.001, max_iter=500, training-loss plateau patience=30, seed 42. Internal validation disabled because splitting after oversampling could place copies of one WSA on both sides. Some fits reach the epoch limit with convergence warnings. |
| Residual network | Width-24 input projection and two residual layers, ReLU and sigmoid output, binary cross-entropy, mini-batch Adam at 0.001, batch=16, L2 alpha=0.001, dropout=0.10, max_epochs=250, seed 42. No internal validation on oversampled rows. Separate diagnostic validation splits original WSAs first. |

Residual optimiser is explicit NumPy project code. Cite/disclose adapted code and AI assistance under institutional rules. Logistic solver iterations are not neural epochs. CPU QA used Python 3.12.14, sklearn 1.9.1, imbalanced-learn 0.14.2 and NumPy 2.5.3. Full tested versions are pinned in `requirements-qa.txt`; execution order and retained artifacts are in README.

### 3.5

Logistic supplies a transparent regularised linear comparator for small tabular data. RF captures nonlinear interactions with leaf-size control. MLP provides a dense neural comparator but risks overfitting 115 development cases. The residual model tests a different tabular architecture with skip connection, dropout and regularisation. Complexity is not assumed to improve results.

## Q4: evaluation and overfitting

### 4.1

Insert the full `reports/qa/evaluation/held_out_metrics.csv` table. Logistic/RF macro-F1 is 0.812, MLP 0.775, residual 0.752. Their low-reliability recalls are 0.625, 0.625, 0.625 and 0.500. AP is average precision; do not describe it as a computed trapezoidal PR-area. All metrics use the same ordered 29 WSAs. These are QA reproduction results on an already disclosed holdout.

### 4.2

Include every `*_classification_report.json`, `*_confusion_matrix.csv/png` and `common_roc.png` from `reports/qa/evaluation/`. Logistic counts are TN=20, FP=1, FN=3, TP=5. A missed low-reliability WSA matters despite high accuracy. `paired_predictions.csv` preserves WSA IDs, labels, predictions and probabilities.

### 4.3

Insert `reports/qa/development/development_comparison.csv` and `validation_comparison.png`. Development mean macro-F1 ranks Logistic 0.800 ± 0.084, RF 0.793 ± 0.071, residual 0.765 ± 0.081, MLP 0.753 ± 0.095. Logistic is champion; RF is strongest remaining comparator. SD represents fold variability, not independent confidence intervals. The historical 0.815/0.833 values belong to an alternate pipeline and cannot be mixed into this comparison.

### 4.4

Include every learning curve and both neural loss/metric history figures/CSVs under `reports/qa/development/`. Mean train-minus-validation macro-F1 gaps are approximately 0.056 (Logistic), 0.187 (RF), 0.220 (MLP), 0.198 (residual). Tree/neural overfitting is material. Controls are regularisation, leaf-size restriction, compact layers, neural weight penalties and residual dropout. The residual diagnostic validation loss flattens/rises while training loss falls. MLP convergence warnings must be disclosed. Diagnostic curves split original development WSAs before fitting preprocessing and resampling and are separate from the full-development model artifacts. No neural settings were retuned after QA test evaluation.

## Q5: champion optimisation

### 5.1–5.2

Tune Logistic only. Fixed grid: C={0.01,0.1,1,10}, L1/L2 through `l1_ratio`={1,0}, liblinear, unchanged preparation, 25 development folds, macro-F1 scoring. The selected C=1/L2 equals the original setting: mean pre/post macro-F1 is 0.800/0.800. No optimisation gain was found. `optimization_search.csv` includes all candidates; `locked_manifest.json` records decisions and data/partition/artifact hashes. Best search-CV score is selection-biased because its folds select parameters; it is not an independent final estimate.

### 5.3

Evaluate the frozen saved pipeline in one QA batch with all comparators. Accuracy=0.862, macro-F1=0.812, ROC-AUC=0.917, AP=0.850; 3 of 8 low-reliability cases were missed. Conditional 95% bootstrap macro-F1 interval is approximately [0.607,0.955]; exact positive-recall interval is [0.245,0.915]. These assume independent cases and do not account for spatial dependence. No practical optimisation gain or fresh untouched-test claim is made.

## Q6: controlled ablation and paired test

### 6.1

Include `reports/qa/development/ablations.csv`. Three one-change experiments retain folds, seeds and estimator settings. Removing the derived-feature group gives macro-F1 delta +0.002; removing scaling −0.004; removing oversampling +0.032. The feature-group ablation removes derived ratios/category together while retaining the six raw features. Feature engineering did not demonstrate a universal gain. Balancing warrants further development investigation, but these results were not used to alter the locked evaluated model.

### 6.2

Compare Logistic with validation-selected RF on identical ordered test cases. Correctness table: both correct=25, Logistic-only=0, RF-only=0, both wrong=4. H0: equal marginal error probabilities; H1: different probabilities; alpha=0.05. Zero discordances give exact two-sided binomial p=1.000; do not reject H0. Nonrejection is not equivalence or proof of fairness/superiority. Use `mcnemar.json` and the case-level CSV. Do not reuse the historical Logistic/Deep-MLP p-value for this new pair.

## Q7: separate deployment handoff

Outside this QA implementation at the user's request; do not mark it complete. The previous app expects the historical artifact/input schema. The corrected local handoff artifact is `models/qa/optimized_champion.joblib`, which accepts six raw numeric predictors and performs feature engineering internally. The team still needs reproducible launch instructions, validation/error handling and three valid plus one invalid demonstration case for assessment.

## Q8: recommendation, reproducibility and team defence

### 8.1

Use the prototype only for investigation with current official information and human judgement. Do not use it alone for funding, penalties or uninterrupted-service assurances. Where observed reliability is available, use the direct rule. Small samples, source inconsistencies, unknown collection periods, prior holdout exposure, geographic dependence and wide recall uncertainty limit conclusions. Urban-dominant test WSAs had 0/2 positives detected, rural-dominant 5/5, unknown composition 0/1; denominators are too small for strong fairness claims but misses must be disclosed. Monitor data quality, drift, calibration, positive recall and subgroup errors; stop use when inputs/outcomes no longer match the validated task.

### 8.2

README gives portable execution order; requirements are pinned. Original data hashes, dictionary, saved assignments, seeds, code, local fitted pipelines and evidence are supplied. Raw CSVs remain unchanged and ignored pending verified redistribution permission. Historical notebooks/artifacts are retained for traceability; corrected execution uses the shared module. Deployment integration remains separate.

### 8.3

The team must provide its presentation and a contribution statement signed by all members, with actual roles and outputs. Update `docs/CONTRIBUTIONS.md` with real evidence and disclose AI assistance under institutional rules. Do not fabricate signatures or participation. Every member should defend the label/leakage exclusions, validation selection, absence of optimisation gain, McNemar nonrejection, subgroup misses and limitations.
