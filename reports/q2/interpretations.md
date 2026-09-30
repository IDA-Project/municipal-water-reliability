# Q2.2 figure interpretations

1. **Target distribution.** There are 39 low-reliability WSAs and 105 reliable WSAs under the pre-declared 60% threshold. This imbalance motivates macro-F1 and training-fold-only resampling.
2. **Missingness.** `Population with access to water` has the highest missingness at 6.2%. Missing predictors are retained and imputed inside training folds; missing targets are excluded from modelling.
3. **Water access and reliability.** The Pearson correlation is 0.76. Access is related to reliability but does not define it; the scatter should be read as association, not causation.
4. **Rural share and reliability.** The Pearson correlation is -0.60. This supports checking geographic and rural/urban disparities, but does not establish that rural composition causes poor reliability.
5. **Urban/rural subgroup view.** Mean reliable supply is 69.6% for the urban measure and 66.7% for the rural measure, compared with 68.0% overall. Aggregate means can hide WSA-level variation and are not fairness proof.
