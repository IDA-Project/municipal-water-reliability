"""Package the frozen QA champion for the app, without retraining or rescoring."""
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]


def package():
    source = ROOT / "models/qa/optimized_champion.joblib"
    manifest = json.loads((ROOT / "reports/qa/development/locked_manifest.json").read_text())
    expected = manifest["artifact_sha256"][source.name]
    if hashlib.sha256(source.read_bytes()).hexdigest() != expected:
        raise ValueError("The champion does not match the locked QA artifact.")
    destination = ROOT / "artifacts/water_reliability_pipeline.joblib"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and hashlib.sha256(destination.read_bytes()).hexdigest() != expected:
        raise FileExistsError("A different packaged model already exists; review before replacing it.")
    shutil.copyfile(source, destination)
    metadata = {
        "sha256": expected, "champion": manifest["champion"],
        "runtime_versions": {name: version(name) for name in ["scikit-learn", "imbalanced-learn", "numpy", "pandas", "joblib"]},
        "data_sha256": manifest["data_sha256"], "assignments_sha256": manifest["assignments_sha256"],
        "classes": {"0": "Reliable supply at least 60%", "1": "Reliable supply below 60%"},
        "scope": "Academic cross-sectional prototype; QA holdout was previously disclosed.",
    }
    destination.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print("Packaged locked QA champion: " + str(destination.relative_to(ROOT)))


if __name__ == "__main__":
    package()
