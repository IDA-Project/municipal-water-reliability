> Historical section retained for traceability. The current full-project answers and consistent experiment are in [QA_CORRECTED_ANSWERS.md](QA_CORRECTED_ANSWERS.md) and [QA_REVIEW.md](QA_REVIEW.md). Do not combine older metrics or scope statements with the corrected QA results.

# Requirements checklist: Questions 1 through 2.4

The assignment PDF is the source of the question wording. The team's answers are standardised to the concrete Q2.3 design: one Water Services Authority (WSA) per row, a cross-sectional binary target, and `Class 1 = reliable supply below 60%`.

| Question | Requirement | Relevant teammate answer | Repository before this work | Added or corrected |
| --- | --- | --- | --- | --- |
| 1.1 | Define the service problem, affected population, decision-maker, benefit, prediction unit, inputs, classes and horizon. | Water-service vulnerability and resource prioritisation. Q2.3 used 144 WSAs and a 60% binary threshold. | Only a provisional project plan; Q1 answers conflicted. | Standardised to WSA-level current-condition classification; removed unsupported ward-level one-month forecasting claims. |
| 1.2 | Name a primary technical metric and service indicator; explain false-positive and false-negative costs. | Macro-F1, prioritisation coverage, wasted response resources, and missed high-need areas. | Proposed metrics only. | Defined macro-F1 plus high-need coverage under a fixed review budget and documented both error costs. |
| 1.3 | Justify relevance, provenance, licence, coverage and limitations; address POPIA, privacy, fairness, misuse and safeguards. | Aggregated public-service data, human review, subgroup checks and decision-support-only use. | Source ownership and terms were unverified. | Corrected the source to DWS NIWIS/WSKS, recorded its academic-use terms, and documented safeguards and limits. |
| 2.1 | Data dictionary and audits of class balance, duplicates, missingness, invalid values, outliers and leakage. | A 13-column dictionary and preliminary pandas checks. | No executable audit. | Added deterministic parsing, target construction, a full audit, national-total reconciliation and machine-readable evidence. |
| 2.2 | At least five purposeful, titled and labelled figures with interpretations. | Target distribution, missingness, access/reliability, rural share/reliability and urban/rural comparison. | Figures existed only inside the project PDF. | Added reproducible figure generation and data-driven interpretation text. |
| 2.3 | Split before learned transformations; justify validation design, stratification/group/time constraints and test isolation. | 80/20 stratified holdout; repeated 5-fold CV on development data; province-grouped sensitivity check; test used once. | No executable split or saved IDs. | Added a fixed seed, saved split assignments, repeated stratified CV and sensitivity-only province grouping. |
| 2.4 | Put cleaning, encoding, scaling, imputation, feature engineering and imbalance treatment in a reproducible, leakage-safe pipeline. | Q2.4 was blank. Q2.3 required fold-local transformations and training-only resampling. | No implementation. | Added safe ratio features, derived categorical encoding, median/mode imputation, robust scaling and random oversampling restricted to training data. |

## Scope boundary

No model from Question 3 is trained here. The Q2.4 preparation pipeline is ready for a Q3 estimator to be appended so every learned step and the sampler remain inside cross-validation training folds.

