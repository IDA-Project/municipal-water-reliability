# Question 3.3: model families

Q3.3 asks for traditional models drawn from the template's examples, plus an MLP and one task-appropriate second neural architecture. The selection is implemented in `src/water_reliability/model_families.py`, which inspects the real estimator classes in `models.build_model_specs()` and writes `reports/q3/q3_3_model_family_compliance.csv` and `reports/q3/q3_3_requirement_checks.json`.

| Model | Estimator class | Family | Template requirement met |
| --- | --- | --- | --- |
| Logistic Regression | `LogisticRegression` | Traditional | Listed traditional example |
| Random Forest | `RandomForestClassifier` | Traditional | Listed traditional example (Decision Tree / Random Forest) |
| Multilayer Perceptron | `MLPClassifier` | Neural | Required MLP |
| Residual Tabular Neural Network | `TabularResidualClassifier` | Neural | Task-appropriate second architecture (tabular network) |

The majority-class baseline is not counted. The data is dense, tabular, non-sequential and non-image, so CNN or GRU/LSTM architectures would not be task-appropriate; this is why a tabular residual network is the second architecture.

## Open item

The template says the second architecture is an "approved tabular neural network". Lecturer approval is not recorded, so the check `second_architecture_approval_recorded` reports OPEN. Once approval exists, run:

```
python -m water_reliability.model_families --approval-evidence "Lecturer email, <date>"
```

and add the date to `docs/DECISIONS.md`.

Hyperparameters, losses, optimiser, epochs, batch size, seeds and environment (Q3.4) and per-model rationale (Q3.5) are separate questions and are not covered here.
