# Decision log

| Date | Decision or question | Owner | Evidence / status |
| --- | --- | --- | --- |
| 2026-09-29 | Begin with local Git repository and project plan | Team | Initial scope requested; no modelling yet |
| 2026-09-29 | Five members; submission due Friday 2 October 2026 | Team | User confirmed; exact cutoff time and presentation slot pending |
| 2026-09-29 | Standardise Q1-Q2.4 on one WSA per row and a binary cross-sectional target: 1 when reliable supply is below 60%, otherwise 0 | Team/user | Confirmed by user; supersedes conflicting ward forecast and three-class drafts |
| 2026-09-29 | Treat the 26 September WSA extract as modelling data and the July national table as a reconciliation control, not extra observations | Team/user | Implemented in the Q2 workflow |
| 2026-09-29 | Exclude WSA name and all reliable-supply counts/percentages from predictors | Team | Leakage control implemented structurally in feature engineering |
| Pending | Obtain lecturer approval for the DWS NIWIS source and 60% team-defined target | TBD | Source terms verified; approval evidence not supplied |
| 2026-09-30 | Train Logistic Regression and Random Forest as the two traditional models, and an MLP plus a residual tabular network as the two neural models | Team | Implemented for Q3.2 on the fixed development split; lecturer approval for the second neural architecture remains to be recorded |
| Pending | Clarify template mark inconsistencies | TBD | Open |
| Pending | Select remote host and repository visibility | TBD | Open |
| 2026-10-04 | Q3.3 model-family compliance check added (`model_families.py`); approval of the residual tabular network still outstanding | Team | `reports/q3/q3_3_requirement_checks.json` reports 7 of 8 checks passing; approval check OPEN |
