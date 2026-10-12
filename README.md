# Municipal Water Supply Reliability in South Africa

IDA117V group project: WSA-level classification of water-supply reliability.

Read the [full QA review](docs/QA_REVIEW.md) and [corrected answers](docs/QA_CORRECTED_ANSWERS.md). Original teammate notebooks remain under `src/` as historical evidence; their Windows paths, alternate model definitions and outputs are not the corrected execution route.

## Task and evidence

One record is one Water Services Authority. Class 1 is reported reliable supply below 60%; class 0 is at least 60%. This team threshold is not an official DWS category. The task classifies cross-sectional conditions, not future interruptions. The export date does not establish the collection period. Predictors are population, water access and urban/rural composition; reliability columns and WSA identifiers are excluded.

The shared model registry contains a majority baseline, Logistic Regression, Random Forest, MLP and a residual tabular neural network. The source route and second neural architecture still need recorded lecturer approval. Macro-F1 selects the champion on 25 development-only CV folds (5 folds, 5 repeats), seed 42. All models share saved Q2 partitions and preprocessing.

The original team already reported test results. Corrected QA results on the same 29 WSAs are reproduction checks, not a newly untouched test. No optimisation gain or statistical superiority was demonstrated. See the QA review for uncertainty and subgroup misses.

## Install

Python 3.12.14 was used for QA. From the repository root on Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-qa.txt
python -m pip check
python -m pytest
```

On Windows activate with `.venv\Scripts\Activate.ps1`. `requirements-qa.txt` pins the tested environment; `requirements.txt` retains broader development installation.

## Data and execution order

Original CSVs are retained unchanged in ignored `data/raw/`. Check SHA256 hashes against [the source register](docs/DATA_SOURCES.md). Verify applicable publisher terms before committing raw copies to a shared repository; redistribution permission is not established by this QA. National categories are reconciliation controls, not training cases.

For a fresh evidence run, from the repository root:

```bash
export MPLCONFIGDIR=/tmp/water-reliability-mpl
export XDG_CACHE_HOME=/tmp/water-reliability-cache
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
python -m water_reliability.workflow
python -m water_reliability.assessment develop --jobs 2
python -m water_reliability.assessment evaluate
python -m pytest
```

Current QA evidence is already saved. Reuse it; do not rerun Q2 over frozen assignments or rescore the completed test run. Q2 was checked in `/tmp/water-qa/q2` to preserve old tracked reports. Development/test commands refuse to overwrite completed results. Any explicitly planned new experiment needs distinct output/artifact paths and honest disclosure that this holdout has been viewed.

Q2 supports `--data` and `--national-data`. Assessment supports `--data`, `--assignments`, `--development`, `--artifacts` and `--evaluation`.

## Outputs and handoff

- `reports/q2/`: original audit, EDA and split assignments, independently checked against uploaded data.
- `reports/qa/development/`: model comparison, fold scores, learning curves, both neural loss/metric histories, search, ablations and locked manifest.
- `reports/qa/evaluation/`: common ROC, classification reports, confusion matrices, paired predictions, intervals, subgroup errors and McNemar.
- `models/qa/`: fitted complete pipelines, retained locally and ignored. `optimized_champion.joblib` is the corrected handoff artifact.
- `docs/`: review, corrected report text, source and contribution records.

The app now loads the corrected complete champion from `artifacts/water_reliability_pipeline.joblib`, with checksum/runtime metadata. Three valid and one invalid UI cases pass. See [deployment readiness](docs/DEPLOYMENT_READINESS.md) for the verified launch and release bundle. The older root-level saved model remains historical evidence and is no longer loaded. No hosted deployment has been performed.

The team must re-export its final report, cite sources/libraries and AI assistance under institutional policy, supply real signed contributions and prepare the presentation. Advisory predictions must not be used alone for allocation, penalties or assurances of reliable service.

## Export review PDFs

Install the optional report tools with `python -m pip install -e ".[report]"`, then run `python scripts/export_qa_reports.py`. The exported review and corrected-answer draft are in `reports/qa/`. They still need team review and actual approval/contribution evidence before submission.

## Run the app

Install `requirements.txt`, then run `python -m streamlit run app.py --server.headless=true --browser.gatherUsageStats=false` from the repository root. Raw source data and training artifacts are not required for prediction.
