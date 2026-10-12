# Deployment readiness — 6 October 2026

The application is prepared for deployment of an **academic prototype**. No hosted deployment or public URL has been created. Lecturer approvals, source terms and contribution evidence remain assessment/governance tasks, as recorded in the QA review.

## Corrected application bundle

- `app.py` passes the six raw population/access counts to the corrected complete saved pipeline. It does not duplicate training feature engineering.
- `artifacts/water_reliability_pipeline.joblib` is a byte-for-byte copy of the locked QA champion, with checksum and required runtime versions in the adjacent JSON file. No retraining or test rescoring occurred during packaging.
- `requirements.txt` installs the pinned `requirements-qa.txt`, including Streamlit and the local package. Use Python 3.12 (QA used 3.12.14).
- The model path resolves relative to the app file. Raw CSVs, ignored `models/qa/`, and analysis reports are not needed at runtime. Keep the packaged artifact and JSON together.
- The historical root-level model remains for traceability; the app no longer loads it. `scripts/package_model.py` reproduces the bundle from the locked local QA champion and refuses to replace a different existing artifact without review.

## Verified behaviour

The full suite passes 45 tests, including three valid population contexts (mixed, urban-only and rural-only), an invalid-input UI case, zero denominators, nonfinite/negative/fractional values, inconsistent total/subgroup access counts, and corrupted artifact detection. Real predictions and probabilities were exercised through Streamlit AppTest. The neural-training suite retains one disclosed convergence warning; it does not prevent application inference.

App errors are friendly. Technical traces and machine paths are logged server-side rather than exposed to the user. Class guidance states the 60% definition, cross-sectional scope and that estimated probability is not a population-reliability percentage or a calibrated guarantee.

## Local launch

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip check
python -m streamlit run app.py --server.headless=true --browser.gatherUsageStats=false
```

On Windows activate `.venv\Scripts\Activate.ps1`. Configure the hosting platform's entrypoint as `app.py` and select Python 3.12. A platform requiring a start command can use `python -m streamlit run app.py --server.address=0.0.0.0 --server.headless=true --browser.gatherUsageStats=false` with its assigned port. Keep the host's normal authentication and TLS configuration; this review does not provision a hosting service.

No data-source credentials or raw records are needed for manual prediction. Do not load a model file uploaded by an end user: joblib artifacts are trusted executable package content, not a user-input format.

## Release checklist

Include `app.py`, `artifacts/` (model plus metadata), `src/water_reliability/`, `pyproject.toml`, and both requirements files. Preserve the original source-data Git policy until applicable redistribution terms are verified. After deployment, verify a real valid prediction and invalid-input error through the hosted UI; a health endpoint alone is not a functional acceptance check. Rerun the existing inference tests if the model, dependencies or UI changes.
