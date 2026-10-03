# Q2.3 - Data splitting and validation design
import numpy as np
import pandas as pd
from sklearn.model_selection import (train_test_split, RepeatedStratifiedKFold,
                                     StratifiedGroupKFold)

SEED = 42
PATH = r"C:\Users\Sekomane\Desktop\Data ScienceTUT\NIWIS_Water Supply Reliability - Population_26-Sep-2026.csv"

# Load the data and turn the text columns
raw = pd.read_csv(PATH).drop(columns="Unnamed: 0")
raw.columns = [" ".join(c.split()) for c in raw.columns]
for c in raw.columns[1:]:
    raw[c] = pd.to_numeric(raw[c].astype(str).str.replace(r"[ %]", "", regex=True),
                           errors="coerce")

# Target: 1 = low reliability (fewer than 60% of people have reliable supply)
y = (raw["% Population Reliable Supply"] < 60).astype(int)
print(f"Rows: {len(raw)} | positives: {y.sum()} ({y.mean():.3f})")

# The file is sorted by province, so rebuild the province from row position
starts = {0: "EC", 15: "FS", 33: "GP", 42: "KZN", 56: "LP", 66: "MP", 83: "NW", 93: "NC", 119: "WC"}
keys = sorted(starts)
province = pd.Series(index=raw.index, dtype=object)
for i, k in enumerate(keys):
    end = keys[i + 1] if i + 1 < len(keys) else len(raw)
    province.iloc[k:end] = starts[k]

# Features: drop the columns that give away the target, plus the WSA name
leak = ["% Population Reliable Supply", "Population Reliable Supply",
        "Urban Population Reliable Supply", "% Urban Population Reliable Supply",
        "Rural Population Reliable Supply", "% Rural Population Reliable Supply"]
X = raw.drop(columns=leak + ["WSA Name"])

# Step 1: split first (stratified 80/20) before learning anything from the data
X_dev, X_test, y_dev, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=SEED)
print(f"dev {len(y_dev)} ({y_dev.sum()} positive) | test {len(y_test)} ({y_test.sum()} positive)")
print(f"overall positive rate {y.mean():.3f} | dev {y_dev.mean():.3f} | test {y_test.mean():.3f}")

# Step 2: validation = repeated stratified 5-fold CV, development set only
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=SEED)
sizes, pos = [], []
for tr, va in cv.split(X_dev, y_dev):
    sizes.append(len(va)); pos.append(int(y_dev.iloc[va].sum()))
print(f"Repeated CV: {len(sizes)} folds | validation size {min(sizes)}-{max(sizes)} | "
      f"positives per validation fold {min(pos)}-{max(pos)}")

# Step 3: sensitivity check with province-grouped folds
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
groups = province.loc[X_dev.index]
print("Province-grouped CV (validation size, positives):")
for i, (tr, va) in enumerate(sgkf.split(X_dev, y_dev, groups), 1):
    print(f"  fold {i}: n={len(va)}, positives={int(y_dev.iloc[va].sum())}, "
          f"provinces={sorted(groups.iloc[va].unique())}")

print("\nProvince x class counts (full data):")
print(pd.crosstab(province, y.rename("low_reliability")))