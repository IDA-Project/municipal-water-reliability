# Reliability calculation and screenshot review

Reviewed 21 October 2026 using the packaged model and the local NIWIS extract.

## Calculation and threshold

- `src/water_reliability/data.py::safe_percentage` calculates `(numerator / denominator) * 100`, returning a missing value for a zero or negative denominator. The reliable-supply audit uses `Population Reliable Supply / Population`, with equivalent urban and rural checks.
- `src/water_reliability/constants.py` defines `TARGET_THRESHOLD = 60.0` and the target source as `% Population Reliable Supply`.
- `src/water_reliability/data.py::clean_raw_data` parses the reported percentage without rounding and sets class 1 when it is strictly below 60, otherwise class 0. Invalid or missing percentages are not labelled. The historical notebooks also use `< 60`.
- `src/water_reliability/features.py::ReliabilityFeatureEngineer` calculates water-access rates and population shares. Water access is a predictor, not a measurement of reliable supply.
- `src/water_reliability/inference.py::predict` obtains the category from the saved model and selects the probability column for class 1 using `model.classes_`. `app.py` formats that probability with one decimal percentage place. This display rounding does not affect classification.

| Observed reliable-supply percentage | Class | Meaning | Verified |
| --- | --- | --- | --- |
| 59.99% | 1 | Unreliable | Yes |
| 60.00% | 0 | Reliable | Yes |
| 60.01% | 0 | Reliable | Yes |

The tests calculate these percentages from counts (59,990, 60,000 and 60,010 out of 100,000) and check the corresponding reported-percentage labels. An additional 59.9999% case remains unreliable, guarding against rounding before comparison.

## Screenshot comparison

| Screenshot | Reproduction inputs in form order | Calculated probability of low reliability | Display | Model category |
| --- | --- | --- | --- | --- |
| First: 83.2% | 100,000; 80,000; 60,000; 50,000; 40,000; 30,000 (app defaults; assumption) | 83.23830174877495% | 83.2% | Unreliable |
| Second: 13.6% | 100,000; 100,000; 60,000; 60,000; 40,000; 40,000 (all visible) | 13.553213561964234% | 13.6% | Reliable |

The first screenshot only shows the last input (30,000). The defaults reproduce its result exactly, but the cropped inputs prevent definitive verification of the original full submission. The second screenshot is fully reproducible.

Both displayed probabilities and categories are consistent with the model. The packaged logistic regression was also checked independently using the transformed inputs, coefficients and intercept: `p(low) = 1 / (1 + exp(-(X @ coefficients + intercept)))`. The scores are 1.6026111818931585 and -1.8529053564944824, giving the probabilities above.

The 60% rule defines the observed supply classes, not a cutoff on the probability of low reliability. The saved binary classifier chooses the higher-probability class (a probability decision boundary of 0.5). A 13.6% probability of low reliability therefore supports a reliable prediction; it does not mean only 13.6% of residents have reliable supply. Its complementary probability is approximately 86.4%, also not an observed supply percentage. The second screenshot has 100% water access, but neither screenshot provides an observed reliable-supply count or percentage.

## Discrepancies and minimal corrections

1. **Misleading success styling.** The first screenshot presents low reliability in a green success box. The app now uses a warning box labelled `UNRELIABLE (below 60%)`. Reliable results retain their success box.
2. **Percentage interpretation.** A caption immediately below the probability now explicitly distinguishes class probability, reliable-supply percentage and water access. The probability calculation and displayed values are unchanged.
3. **Source rounding at the threshold.** All comparable overall (144), urban (142) and rural (137) count-derived percentages are within the existing 0.5 percentage-point tolerance of the reported percentages. However, Nkomazi has 268,409 people with reliable supply out of 450,134: **59.62868834613693%**, which implies Unreliable. Its source reports **60%**, which implies Reliable. The old audit missed this class disagreement because the difference was within its rounding tolerance. The audit now flags a threshold disagreement regardless of that tolerance. The supplied extract has one such finding, in addition to its existing 57 component/total findings.

The declared training target uses the publisher's reported percentage, so Nkomazi remains class 0 under that definition. The new audit flag does not silently relabel it. Adopting a count-derived target would require a separately reviewed target change and model revalidation; the current model estimates reported-percentage classes and cannot resolve that source precision limitation.

## Validation and scope

`python -m pytest tests/test_data_and_audit.py tests/test_inference_and_app.py tests/test_real_data_contract.py -q` passes **29 tests**, including real pipeline inference, Streamlit display checks, boundary arithmetic and labels, and the rounding-disagreement regression. `git diff --check` passes.

No model was retrained. Raw data, target construction, saved models, metadata, training code and frozen evaluation reports were preserved. The packaged model SHA256 remains `ec191f3cc33eae4ce4a8e4c7c83c4866ffd8d8c88a1630055681db7d49965f28`.
