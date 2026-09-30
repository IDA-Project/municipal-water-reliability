# Municipal Water Supply Reliability in South Africa

> Status update (29 September 2026): Questions 1 through 2.4 are now standardised and implemented. See `REQUIREMENTS_CHECKLIST.md`, `QUESTIONS_1_TO_2_4.md`, and `reports/q2/` for current evidence. This file retains the original team plan and later-question roadmap.

## Project plan

Prepared 29 September 2026. Course: IDA117V, semester 2, 2026.

This plan organises the work needed to investigate municipal water-supply vulnerability and deliver the course project. The team should first assign owners, confirm the data and target, and agree on the evaluation design. Analysis, model results and final recommendations remain team tasks.

Team size: **five members**. Submission deadline: **Friday 2 October 2026**, confirmed by the team; exact time to be confirmed. Names, student numbers, group number, lecturer and presentation slot remain unassigned. The template lists presentations from **5 to 17 October 2026**; confirm the team's separate presentation slot.

## Purpose and proposed scope

Unreliable municipal water supply can cause repeated or prolonged interruptions and reliance on stored water, tankers, boreholes, vendors or rivers. The proposed stakeholder is a municipal water-services planning team, with DWS as a possible secondary audience. The intended decision is where to prioritise further investigation and service-improvement resources. The team must confirm the stakeholder and support the problem statement with cited evidence.

The proposed classification is **1 = vulnerable to unreliable water supply** and **0 = not vulnerable**. These are proposed class meanings, not existing labels in the supplied files. Define a defensible operational rule before training. The current data suggests a Water Services Authority (WSA) as the initial unit, subject to geographic verification; do not assume each row represents a local municipality or household.

A single aggregate snapshot can support a study of current area-level vulnerability. A claim about future interruptions requires dated historical predictors and a later outcome. Confirm the prediction horizon explicitly. Area-level findings must not be presented as household-level predictions.

Candidate primary technical metric: macro-F1, subject to class balance and stakeholder costs. Also examine vulnerable-class recall and precision. A missed vulnerable area could delay attention; a false alert could divert limited assessment resources. A proposed service indicator is the proportion of independently identified vulnerable areas captured within a fixed inspection budget. Define that budget and measure before evaluation; model accuracy alone does not demonstrate improved water delivery.

## What the supplied data supports

The following is a structural inventory, not a completed data audit. Source filenames are retained in `data/raw/`.

| File | Observed structure | Planned use |
| --- | --- | --- |
| `NIWIS_Water Supply Reliability - Population_26-Sep-2026.csv` | 144 records; 144 distinct WSA names; 14 columns including an unnamed leading column. Population, access and reliable-supply measures for total, urban and rural populations, with percentages. | Candidate area-level starting dataset. Verify authority boundaries, coverage, measurement definitions and dates. |
| `Water Supply Reliability  - Population - National[Jul-2026].csv` | 9 category/value records in 2 columns. National total, urban and rural population/access/reliable-supply counts. | Context and possible reconciliation after verifying matching definitions and periods. These are aggregate categories, not nine independent training cases. |

The filenames suggest export/report dates but do not establish collection periods. Record exact source URLs, owner, licence or reuse terms, collection period, download date and methodology. Those details are currently unverified. Preserve the original files unchanged.

The WSA file contains space-separated numbers and percentage strings. The national file has leading apostrophes in labels. Audit parsing, missingness, duplicates, ranges, population identities and percentage denominators. Confirm whether reliability percentages refer to all population or only people with access; do not treat population measures as household measures.

Household counts, interruption frequency/duration, Blue Drop scores, population density, historical records and an explicit province/code field are not supplied as dedicated columns. The requested feature list is a research wish list, not the current schema.

Prioritise additional data only where it resolves a defined gap. Census/demographic or geographic data could provide household counts, area and geographic keys. GHS may offer interruption measures, subject to its survey geography and weights. Blue Drop may offer technical indicators if its assessment units and periods can be reconciled. Rainfall, drought, temperature and resource conditions are optional extensions after core feasibility is established. No additional dataset is assumed available or approved.

## Decisions required before modelling

1. **Source approval.** Prepare the template's one-page concept note, dataset card and feasibility check. Obtain written lecturer approval for the source route where required by the template. Record approval in the decision log.
2. **Geography and joins.** Confirm what a WSA covers, identify stable codes and document any municipality-to-WSA crosswalk. Check boundary changes, overlapping authorities and join cardinality. National aggregates must not be appended as area observations or counted as extra sample size.
3. **Target.** Document the reliability definition, denominator, vulnerability threshold and stakeholder rationale. Do not select a threshold to maximise test performance or merely force balanced classes. If only a rule-derived target is feasible, be explicit about what the model can add beyond that rule.
4. **Leakage.** If the target is derived from reliable-supply percentage, exclude that percentage and contemporaneous reliable-supply counts, urban/rural components, ratios or other fields that reconstruct it. Evaluate remaining predictors for availability at the intended prediction time. Historical reliability is eligible only with a valid earlier timestamp.
5. **Feasibility.** Assess independent sample size and class counts after cleaning. The 144 starting records may constrain neural modelling and statistical power. Never inflate sample size by duplicating rows or treating total/urban/rural measures as independent areas. Escalate an infeasible design to the lecturer and seek compatible data or revised scope.
6. **Evaluation contract.** Choose split proportions/folds after checking class and group counts. Use a shared stratified design where appropriate, with group or time constraints where needed. Freeze IDs, seed, scoring rule and comparator selection before final evaluation.

## Work allocation

Assign the five packages below to the five team members. Each member should also review another member's work. Names remain open for the team to allocate; all members must understand and defend the complete pipeline.

| Member and package | Owner / reviewer | Deliverables | Dependencies |
| --- | --- | --- | --- |
| 1 Problem and report coordination | TBD / TBD | Concept note; stakeholder/target rationale; source approval and lecturer queries; Q1 and Q8; report assembly, slides and contribution statement coordination. | Starts immediately; agree target with 2 and 5. |
| 2 Data and preparation | TBD / TBD | Source register and dataset card; geography checks; dictionary/audit; reproducible cleaning and preprocessing; shared data contract; Q2.1 and Q2.3–2.4. | Source access and approved target; freeze split with 5. |
| 3 Exploration and traditional models | TBD / TBD | Five required EDA figures and interpretations; naive baseline plus two traditional models; configurations, predictions and model-specific curves; Q2.2 and traditional Q3/Q4. | Audited development data and common split/pipeline. |
| 4 Neural models | TBD / TBD | MLP plus approved task-appropriate second neural architecture; configurations, predictions, learning and loss/metric curves; neural Q3/Q4. | Feasible sample/classes and shared data contract. |
| 5 Evaluation and interface integration | TBD / TBD | Shared evaluation protocol and comparison outputs; saved-pipeline UI, validation and launch guide; paired McNemar calculation; Q4, Q6.2 and Q7 integration. | Agree protocol early; final integration follows model handoff. |

The owner of the validation-selected champion (member 3 or 4) performs Q5 tuning and leads the three Q6.1 ablations with member 5. Member 1 helps member 3 format EDA evidence and member 5 document/demo the interface. Each member writes their own report sections; member 1 integrates them. All five review the final evidence, rehearse and sign a truthful contribution statement.

## Evidence checklist aligned to the template

| Report section | Work to complete |
| --- | --- |
| Q1 Problem and governance | Define affected population, stakeholder, prediction unit/horizon, inputs/classes, metric and service indicator. Document provenance, licence, coverage, privacy, fairness and misuse safeguards. |
| Q2 Data and pipeline | Dictionary; class balance, duplicates, missingness, invalid values, outliers and leakage audit. At least five figures: target distribution, quality/missingness, two feature–target relationships, and one geographic/temporal/subgroup view. Explain each figure. Split before fitting transformations; keep preprocessing within training folds. |
| Q3 Baseline and models | Naive baseline plus at least four substantive models: two traditional and two neural. Candidate traditional models: logistic regression and random forest. Neural models: MLP and a lecturer-approved tabular architecture if the data remains tabular. Do not force a CNN/LSTM onto non-image/non-sequence data. Record architectures, losses, optimiser, epochs, batch size, regularisation, early stopping, seeds and environment. |
| Q4 Evaluation | Identical held-out cases and comparable metrics: accuracy and macro-F1, plus relevant precision, recall, weighted-F1, ROC-AUC and PR-AUC (or justify omission). Classification reports and confusion matrices per model; common ROC plot; comparison table/bar chart; validation variability where feasible. Learning curves and train–validation gaps for every substantive model; loss/metric histories for neural models. |
| Q5 Optimisation | Select champion using validation only. Tune only that model using a documented search space and scoring rule. Compare pre/post optimisation; lock the pipeline and evaluate once in the final test batch. |
| Q6 Ablation and paired test | At least three controlled one-change ablations under fixed splits/seeds/protocol. Select meaningful components once the pipeline exists. Compare locked champion and strongest validation-selected comparator on the same held-out cases using McNemar: correctness 2×2 table, H0/H1, alpha 0.05, statistic/exact p-value and practical interpretation. Use exact binomial form for small discordant counts. |
| Q7 Interface | Working UI loads complete saved preprocessing/model pipeline, accepts valid inputs, shows class and probability where appropriate, gives plain-language guidance, validates inputs and handles errors. Include limitations, launch instructions and three valid cases plus one invalid case. |
| Q8 Handover | Evidence-based recommendations, uncertainty and fairness discussion; reproducible repository with environment, execution order and source links; saved pipeline and UI; presentation/demo and signed contribution statement. |

Develop and select models using training/validation data. Pre-register ablations and final comparisons before revealing test results. Generate all required final model reports in one planned evaluation batch on the same held-out IDs; do not repeatedly inspect test results to change models, labels or features. Retain paired predictions for McNemar and report limited power honestly if the test set is small.

The supplied template contains inconsistent marks: Q3 is headed 18 while its subparts/rubric total 22; Q7 is headed 10 while its subparts/rubric total 7; question heading totals sum to 99 while the rubric totals 100. Ask the lecturer which weights apply. Plan to satisfy every listed requirement regardless.

## Provisional schedule

The confirmed submission deadline is Friday 2 October 2026. Confirm the cutoff time immediately. The schedule below is compressed and depends on rapid source approval and data feasibility. Keep optional environmental enrichment outside the critical path. If the required evidence cannot be produced credibly, contact the lecturer promptly rather than fabricate results or omit requirements silently.

| Date | Parallel work and milestone | Acceptance condition |
| --- | --- | --- |
| Tuesday 29 September | Allocate five roles; member 1 prepares concept/approval questions; member 2 checks sources and target feasibility; members 3–5 agree modelling/evaluation handoffs. | Owners assigned; exact cutoff queried; source and target gaps identified; lecturer questions sent by the team. |
| Wednesday 30 September | Freeze approved target, safe features and shared splits as soon as feasible. Member 2 builds pipeline; member 3 starts EDA/traditional models; member 4 neural models; member 5 shared evaluation and UI scaffold; member 1 drafts context/report. | Approval where required; usable classes; common input schema and splits; baseline and model runs underway. |
| Thursday 1 October | Finish compliant 2+2 models and validation evidence; select champion. Champion owner tunes and runs three ablations; member 5 integrates pipeline/UI; other members assemble report and review. | Locked models/comparator and final evaluation protocol; evidence gaps identified before test access. |
| Friday 2 October before cutoff | Run planned locked test batch and McNemar; finish report and package; verify clean reproduction and three valid/one invalid UI cases; all five review and submit with a time buffer. | Required report, code, data documentation, saved pipeline, UI instructions, evidence and contribution statement ready; submission receipt retained. |
| Confirmed presentation slot | Rehearse and present within the template's 5–17 October window, unless lecturer confirms otherwise. | Every member participates; working live or recorded demo; preserve integrity of submitted test evidence. |

The template suggests a 15–20 page technical report excluding cover/references/appendices and a 10–12 minute presentation plus questions. Confirm actual submission instructions with the lecturer.

## Git workflow and handover

Use `main` for reviewed work and short branches such as `data/audit` or `models/traditional`. Commit small coherent changes, record issue/task IDs where available, and have another member review before merging. Choose one owner for shared split definitions and one for report integration to reduce conflicting edits.

Keep raw inputs unchanged. Track scripts, notebooks, plans and source metadata. Local source copies are excluded from Git until redistribution terms are verified; share authorised acquisition instructions instead. Ignore credentials, environments, generated datasets and trained artifacts by default. Arrange an approved artifact delivery route for the final saved pipeline so the submission remains runnable.

Record every experiment's data version, split IDs, seed, model configuration, validation metrics and artifact locations. Add exact dependency versions and execution commands when the implementation exists. Do not claim the initial scaffold is an executable ML solution.

## First team meeting checklist

- Assign the five members and reviewers to packages 1–5; fill in student/group details and confirm Friday's cutoff time.
- Confirm WSA versus municipality scope, current versus future prediction, target definition and stakeholder.
- Assign source URL/licence/metadata verification and request lecturer approval/mark clarification.
- Agree on feasible enrichment, the second neural architecture and the evaluation contract.
- Choose a remote Git host and repository visibility, set up contributor Git identities, then share the repository.

## Planning sources

- User-provided topic, problem statement, proposed target and candidate feature/dataset list.
- `IDA117V Project Template.pdf`, supplied by the user: scope and approval checkpoint pp. 1–4; assessment questions pp. 5–7; evidence/submission requirements pp. 8–9; rubric pp. 10–13; suggested roles p. 14.
- The two supplied CSVs listed above, inspected for schema and record counts on 29 September 2026. A full quality audit and original source verification are pending.

The template supplies course requirements to plan for; it does not authorise completing or publishing the full assessed project at this stage.
