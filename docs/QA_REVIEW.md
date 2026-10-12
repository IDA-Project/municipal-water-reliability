# Full project QA — 6 October 2026

> Follow-up code recheck: the app has now been integrated with the corrected packaged pipeline. Three valid and one invalid UI cases pass, as does a clean installation without raw data or training files. See [DEPLOYMENT_READINESS.md](DEPLOYMENT_READINESS.md). Earlier statements below about the app being unchanged describe the original QA scope; no hosted deployment has occurred.

## Overall finding

The supplied answer sheet is not submission-ready. It contains meaningful work, but individually written sections describe incompatible tasks and experiments. Marks are at risk for missing evidence, an inconsistent second neural architecture, a comparator selected contrary to the stated validation ranking, and unsupported provenance/forecasting claims. This review cannot determine the lecturer's exact deductions or guarantee a grade.

The 27-page `IDA117V Project Template (1).pdf` was treated as the original brief. It already contains draft answers. The 35-page `IDA117V Project Template (1) (1).pdf` was treated as the team's working answer sheet. Assessment question wording and the rubric were distinguished from annotations and answers; suggestions inside either document were not treated as instructions to change the task. Page references below refer to the 35-page answer sheet unless explicitly labelled original.

The repository was reviewed on `work` after the QA merge. Deployment work was excluded as requested. Q7 remains an assessment requirement, even though its completion is outside this QA change.

## Most consequential discrepancies

1. **Three incompatible Q1 tasks.** Pages 5–6 describe wards/supply zones, two interruption events and a one-month forecast. Pages 31–35 append a municipality-level, three-class Stats SA proposal. Q2 and the implementation use a WSA-level binary NIWIS rule. Use only the implemented task throughout: one WSA, population/access inputs, low reliability below 60%, cross-sectional conditions. Do not claim household prediction, interruption forecasting or pipe-failure localisation.
2. **Unsupported source and licence claims.** Page 6 says Stats SA census plus asset registers and CC BY. The actual files are DWS NIWIS exports. No asset-register inputs are present. CC BY is not evidenced. The exact observation period and written source approval are unrecorded. The source register has a DWS terms summary, but the current network returned a proxy CONNECT 403 when rechecking the source page. Do not substitute a guessed licence or claim publication permission has been verified.
3. **Two model sets and two preparation workflows.** Q3 source code uses Logistic Regression, Random Forest, MLP and a residual tabular network, with one-hot encoding and training-only oversampling. Later notebooks use MLP plus Deep MLP, omit the categorical feature and oversampling, and use different estimator settings. Making an MLP deeper does not independently establish a distinct approved second architecture. Numbers from these workflows must not be combined into a single experiment.
4. **Wrong historical McNemar comparator.** Page 19 ranks validation macro-F1 as Logistic 0.815, MLP 0.796, Random Forest 0.795, Deep MLP 0.785. Page 23 nevertheless calls Deep MLP the strongest comparator. Under that ranking, MLP is the strongest remaining comparator. The corrected consistent QA experiment ranks Random Forest second. Its paired predictions are used in the new test; the historical p-value cannot be carried over to a different pair.
5. **Evidence claimed but not supplied.** Pages 18–19 describe classification reports, confusion matrices, common ROC curves, learning curves and neural histories without actually including those results/figures. Only a subset of Q4 code and saved outputs was committed. Paragraphs saying evidence was generated are not substitutes for the evidence.
6. **The test has already been disclosed.** Q4, Q5.3 and Q6.2 notebooks each score the same partition. It is possible to compute all final comparisons in one planned batch, but the existing history does not establish such a lock. It does not prove that tuning used test labels, either. Report what happened honestly. QA reruns on this split are reproduction checks, not fresh untouched-test estimates. Do not silently invent a new split to improve scores.
7. **Promised uncertainty is absent from the old submission.** Page 13 says bootstrap intervals and threshold sensitivity are reported; no corresponding results are supplied. The QA now includes conditional bootstrap and recall intervals. A threshold-sensitivity experiment has not been run; remove the assertion that it has.
8. **Internal neural validation was optimistic.** In the Q3 pipelines, oversampling preceded the neural estimator's internal early-stopping split, allowing duplicated WSAs on both sides. External CV rows remained separate, but internal stopping evidence could be optimistic. The corrected registry disables this internal validation. Separate diagnostic curves split original development WSAs before fitting preprocessing and oversampling.

## Question-by-question assessment

“At risk” means evidence or reasoning needs correction; it does not mean a known numerical penalty. The marks column uses subquestion marks printed in the brief.

| Question | Marks | Original answer/evidence | QA finding and required correction |
| --- | ---: | --- | --- |
| 1.1 | 4 | Pages 5–6 and 31–35 give conflicting units, targets, features and horizons. | At risk. Replace with the WSA binary NIWIS task; collection period is unverified. |
| 1.2 | 2 | Macro-F1 and error costs are reasonable; “failures mitigated” is unmeasured. | Partly aligned. Separate a proposed operational success measure from measured model metrics. |
| 1.3 | 4 | Incorrect CC BY/asset-register claims; privacy/fairness safeguards are proposed. | At risk. Correct publisher, cite actual terms, record source approval, report actual coverage and limits. Aggregation alone does not certify POPIA compliance. |
| 2.1 | 4 | Dictionary and code snippets exist; PDF code is not a clean executable audit. | Repository audit now reproduced. Include 105/39 class counts, 45 missing cells, 57 inconsistency flags across 22 WSAs, and excluded leakage columns. |
| 2.2 | 6 | Five EDA figures exist, with interpretations; some target discussion is continuous rather than binary. | Mostly aligned. Use reproducible binary target plot and labelled figures in `reports/q2/figures/`; describe source anomalies and noncausal associations. |
| 2.3 | 3 | 80/20 split, repeated CV, province sensitivity; later code changes five repeats to three. | At risk from inconsistency. Shared QA runner uses saved IDs and 25 folds. Province mapping is inferred from row order, not a verified geographic key. The existing grouped output counts cases; it does not establish model performance in unseen provinces. |
| 2.4 | 5 | Pages 14–16 describe encoding, scaling, imputation and oversampling. | Aligned with Q2/Q3 package, not old later notebooks. Corrected runner uses the same fold-local pipeline everywhere. |
| 3.1 | 2 | Page 16 explains a baseline but does not give its actual fitted result. | Include majority baseline validation macro-F1 0.422 and zero low-reliability recall. It is excluded from four substantive models. |
| 3.2 | 10 | Page 17 has question wording without a complete answer; training exists in code. | Include four models, common folds, fitted artifacts and configuration evidence from the QA experiment. |
| 3.3 | 3 | Residual network in package; Deep MLP in later work. | At risk. Use residual tabular network consistently; written lecturer approval is still missing and cannot be manufactured. |
| 3.4 | 5 | Page 18 leaves this answer blank; settings are scattered in code/notebooks. | Include architecture, loss, optimiser, epochs, batch size, regularisation, stopping, seeds, exact environment and execution commands. See corrected answers. |
| 3.5 | 2 | Page 18 leaves the answer blank. | Include linear transparency, nonlinear trees, dense MLP, and residual tabular rationale without claiming complexity guarantees improvement. |
| 4.1 | 4 | Narrative metrics are present; old evaluated model set differs from Q3. | Replace the complete table with consistent QA metrics. Label AP accurately rather than imply trapezoidal PR-AUC was calculated. |
| 4.2 | 5 | Reports/matrices/ROC are described, not attached. | QA now supplies classification JSON, confusion CSV/PNG and one common ROC figure for every model. Include them in report/appendix. |
| 4.3 | 3 | Validation selection of Logistic is sensible; comparison table/bar chart are missing. | QA supplies means, SDs, per-fold scores and chart. SD of dependent repeated folds is not a confidence interval. Selection never uses held-out ranking. |
| 4.4 | 4 | Some training/validation numbers, but missing curves and incomplete validation-loss evidence. | QA supplies learning curves, fold gaps and both loss/metric curves for two neural models. Fixed diagnostic development split differs from full-development fitting and is labelled. MLP convergence warnings must be disclosed. |
| 5.1 | 4 | Defined search and scoring exist, but use the alternate pipeline. | Corrected development search tunes only the validation champion; search CSV and locked parameters are saved. |
| 5.2 | 4 | Claimed increase 0.815→0.833 mixes an alternate experiment with Q3. | Do not reuse that improvement in the corrected experiment. Corrected search gives 0.800→0.800. Best search-CV estimate is selection-biased; no independent optimisation gain is established. |
| 5.3 | 2 | Locked model reported, but recomputed independently in later notebooks. | QA evaluates saved frozen artifacts once and preserves paired predictions. Disclose prior test exposure; report small sample and uncertainty. |
| 6.1 | 5 | Three ablations are described and notebook outputs exist. | Old results belong to the alternate pipeline. QA uses three one-component ablations with fixed folds/seed and saves mean/SD/deltas. Do not tune the final model after observing ablations/test. |
| 6.2 | 5 | Exact p=1 for a plausible single discordance; wrong comparator selected. | Exact-test arithmetic is not the main issue; model-pair selection is. QA uses validation-selected RF and gives zero discordances, p=1. Nonrejection does not prove equivalence. |
| 7.1–7.4 | 7 total | Answer sheet blank; Streamlit source and older saved model exist. | Excluded from implementation by user. Still required for assessment. Existing model excludes feature engineering, Streamlit is undeclared, launch evidence is missing and model version differs from this environment. Separate deployment handoff remains needed. |
| 8.1 | 2 | Page 24 gives cautious recommendations; no actual subgroup findings. | Update metrics and include observed subgroup errors and wide uncertainty. Avoid claiming resource-allocation benefit was demonstrated. |
| 8.2 | 3 | Windows paths, missing source files/model artifacts, incomplete execution order. | QA adds portable runner, exact environment, original local data and evidence. Keep originals unchanged; confirm redistribution permission before publishing raw copies. |
| 8.3 | 2 | Presentation/contribution question blank; contribution log is empty. | Team must supply actual roles, signatures and presentation evidence. Individual sections are allowed, but shared understanding and truthful contribution evidence are required. |

## Verified data and QA results

Both uploaded source hashes match `docs/DATA_SOURCES.md`. The WSA source has 144 unique WSAs, 105 class-0 and 39 class-1 records. Fixed saved assignments contain 115 development cases (84/31) and 29 test cases (21/8). The national table has nine control categories, not nine extra model observations. Eight controls reconcile; access population differs by 552,511 alongside nine missing WSA values. Source anomalies are preserved and flagged, not silently repaired.

The corrected model registry is Logistic Regression, Random Forest, MLP and residual tabular network. All use the same raw-safe feature selection, imputation, robust scaling, categorical encoding and training-only oversampling. The target source and related reliable-supply columns are unavailable to feature engineering. WSA names are IDs only. Seed is 42. Development validation uses five folds repeated five times. Neural stopping uses training loss/fixed epoch limits; separate raw development splits provide honest diagnostic validation curves.

| Model | Development macro-F1 mean ± SD | Train macro-F1 mean | QA test accuracy | QA test macro-F1 | Low-reliability recall | ROC-AUC | AP |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Majority baseline | 0.422 ± 0.006 | 0.422 | 0.724 | 0.420 | 0.000 | 0.500 | 0.276 |
| Logistic Regression | 0.800 ± 0.084 | 0.857 | 0.862 | 0.812 | 0.625 | 0.917 | 0.850 |
| Random Forest | 0.793 ± 0.071 | 0.980 | 0.862 | 0.812 | 0.625 | 0.923 | 0.842 |
| MLP | 0.753 ± 0.095 | 0.972 | 0.828 | 0.775 | 0.625 | 0.887 | 0.837 |
| Residual tabular network | 0.765 ± 0.081 | 0.962 | 0.828 | 0.752 | 0.500 | 0.869 | 0.778 |

These QA numbers replace, rather than confirm, the alternate-pipeline claims in the PDF. Lower neural validation scores after removing duplicated internal-validation samples are reported honestly. They do not show that every difference was caused solely by that correction.

The existing root-level saved Logistic artifact was also checked separately with its original 11-feature input schema. It reproduces the PDF's historical final values: accuracy 0.862, macro-F1 0.812, ROC-AUC 0.905 and AP 0.834. This confirms that those particular numbers have supporting model evidence; the objection is combining them with a different Q2/Q3 pipeline. The artifact raises sklearn version warnings (saved with 1.9.0; QA uses 1.9.1) and is retained as historical evidence, not substituted for the corrected full pipeline.

The fixed Logistic search evaluates C in {0.01, 0.1, 1, 10} and L1/L2 through `l1_ratio` {1,0}, with liblinear and unchanged preprocessing. The strongest validation configuration remains C=1, L2; mean macro-F1 is 0.800 before and after. Optimisation need not improve results to be a valid experiment. Do not claim a gain merely because a search was performed.

Three fixed-fold ablations give: remove derived-feature group, +0.002 macro-F1; remove scaling, −0.004; remove oversampling, +0.032. This suggests the balancing choice warrants future development investigation. It does not justify changing the locked pipeline after the reported test or claiming causal/general population gains from one small dataset.

QA McNemar table (rows: Logistic correct/wrong; columns: RF correct/wrong):

| | RF correct | RF wrong |
| --- | ---: | ---: |
| Logistic correct | 25 | 0 |
| Logistic wrong | 0 | 4 |

Exact two-sided p=1.000, alpha=0.05. There is no evidence of different error rates for this pair on these 29 cases. The small sample and zero discordances prevent a useful superiority claim; equality is not established.

The champion confusion matrix has TN=20, FP=1, FN=3, TP=5. Macro-F1's stratified 500-resample conditional bootstrap interval is approximately [0.607, 0.955]; low-reliability recall's exact 95% binomial interval is [0.245, 0.915]. These assume independent cases and do not account for spatial dependence. The group check shows urban-dominant WSAs had 0/2 positive cases detected, rural-dominant 5/5, and unknown composition 0/1. These denominators are too small for strong fairness conclusions, but the misses must be disclosed. Sensitive attributes were not audited, and absence of such features does not prove fairness.

## Marking and penalty interpretation

The rubric gives marks for verifiable evidence and reasoning. Missing sections can earn little or no credit even when another teammate has working code. The clearest historical risks are Q1 coherence/provenance, Q3.3–3.5, Q4 missing visual evidence, Q6.2 comparator selection and Q8 reproducibility/contribution. The corrected evidence addresses technical gaps; it does not supply missing lecturer approval, signatures or a final edited report.

There is no stated penalty merely for answering individually as a group. The brief explicitly allows roles to overlap but says every member must understand and defend the complete pipeline; individual marks may be moderated where contribution evidence differs. Do not invent names, signatures or work claims.

The template has inconsistent mark totals: Q3 heading 18 versus subparts/rubric 22; Q7 heading 10 versus subparts/rubric 7. Question headings total 99, while rubric totals 100. Ask the lecturer which weighting controls; this review uses the printed subquestion requirements without inventing a corrected rubric.

## Remaining team actions

1. Obtain/document lecturer approval for the DWS source route and residual tabular architecture; confirm mark weights.
2. Replace contradictory PDF answers with `docs/QA_CORRECTED_ANSWERS.md`, insert the supplied tables/figures and re-export a clean report. Keep evidence status clear and retain original PDFs for traceability.
3. Confirm source collection period, reliable-supply denominator and the 60% threshold's stakeholder rationale. It is a team cutoff, not an official DWS category. Because the observed reliability percentage defines the label, directly applying that rule is preferable whenever the percentage is already available; ML adds value only when estimating the category from available independent predictors.
4. Verify publisher terms before including raw CSVs in a public/shared commit. They are currently available locally and ignored; no raw copy has been published by this QA.
5. Add real contribution records, signed statement, references and AI-assistance disclosure under institutional policy; prepare the team presentation.
6. Complete the separate UI/deployment handoff and required three-valid/one-invalid demonstration. The original app/artifact were left unchanged.

## Reproduction evidence

Use `README.md` and `requirements-qa.txt`. Q2 was rerun into an isolated directory to preserve the team's old evidence; Q4–Q6 outputs are in `reports/qa/development/` and `reports/qa/evaluation/`. Fitted full preprocessing/model artifacts are in ignored `models/qa/`. The locked manifest hashes data, saved assignments and fitted artifacts. A second test evaluation into the same destination is refused; use the recorded predictions for interpretation and report formatting.

Historical notebooks under `src/` are retained as original evidence. Their Windows-specific paths and alternate model definitions are not the corrected execution route; do not run them to regenerate the submission. No original notebook output was silently rewritten to look like the corrected experiment.

Validation completed: 28 automated tests passed with no skips; one expected MLP iteration-limit warning remains and is disclosed. Dependency integrity (`pip check`) and diff whitespace checks passed. The reproduced Q2 audit matches the saved audit exactly; EDA correlations agree within 5e-16 floating-point tolerance. Existing completed-development and test-evaluation locks were exercised and correctly refused repeat scoring. The residual network's analytical gradients were checked against finite differences. No deployment was run or changed, no Git commit/push was made, and raw source files remain ignored.
