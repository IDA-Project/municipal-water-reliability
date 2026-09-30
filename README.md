# Municipal Water Supply Reliability in South Africa

IDA117V group project investigating WSA-level vulnerability to unreliable water supply.

Questions 1 through 2.4 are documented and implemented as a reproducible data-audit, visualisation, split and preparation workflow. Start with the [requirements checklist](docs/REQUIREMENTS_CHECKLIST.md) and [written answers](docs/QUESTIONS_1_TO_2_4.md).

## Agreed task

- Unit: one Water Services Authority (WSA).
- Target: `1 = low reliability` when the reported population reliable-supply percentage is below 60%; `0 = reliable` otherwise.
- Horizon: the supplied September 2026 cross-sectional snapshot, not a future forecast.
- Primary metric for later modelling: macro-F1.

## Setup and execution

Use Python 3.10 or newer. From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m water_reliability.workflow
python -m pytest
```

The workflow reads the two unchanged files in `data/raw/`. Raw source data and fitted artifacts are intentionally ignored by Git because the DWS terms restrict redistribution. Alternative paths can be supplied with `--data` and `--national-data`.

Generated Q2 evidence is written to `reports/q2/`: the audit, dictionary, national reconciliation, five figures and interpretations, fixed split assignments, and the Q2.4 pipeline summary. The fitted development-only preprocessor is written to `models/q2_preprocessor.joblib`.

## Repository layout

- `docs/PROJECT_PLAN.md` — scope, work allocation, milestones and assessment checklist.
- `docs/` — source register, decisions and contribution records.
- `data/raw/` — unchanged local source copies, excluded from Git.
- `data/processed/` — generated datasets, excluded from Git.
- `references/` — local course template, excluded from Git.
- `src/water_reliability/` — reproducible Q2 preparation and evidence code.
- `tests/` — automated cleaning, audit, leakage, split and workflow checks.
- `notebooks/` — optional exploration notebooks.
- `app/` — future interface code.
- `reports/q2/` — generated audit, figures and split evidence.
- `models/` — future saved pipelines, excluded from Git by default.

## Collaboration

The repository has `main`, `Dev`, `QA` and `Prod` branches; this work remains on the current `Dev` branch. Request a teammate's review before merging. Keep raw sources unchanged and excluded from shared version control under the recorded DWS terms. Do not commit credentials or personal records.

Record decisions in [docs/DECISIONS.md](docs/DECISIONS.md) and actual contributions in [docs/CONTRIBUTIONS.md](docs/CONTRIBUTIONS.md). Review [docs/DATA_SOURCES.md](docs/DATA_SOURCES.md) before data work.
