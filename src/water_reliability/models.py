"""Model definitions for Questions 3.1 and 3.2."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline as ImbalancedPipeline
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from .constants import RANDOM_STATE
from .features import ReliabilityFeatureEngineer
from .pipeline import build_column_preprocessor


class TabularResidualClassifier(ClassifierMixin, BaseEstimator):
    """A compact residual neural network for dense tabular binary data.

    The network uses an input projection, a two-layer residual block and a
    sigmoid output. It trains with mini-batch Adam, L2 weight decay, dropout
    and an internal stratified early-stopping split. The estimator follows the
    scikit-learn API so preprocessing and resampling stay inside each external
    cross-validation training fold.
    """

    def __init__(
        self,
        hidden_dim: int = 24,
        learning_rate: float = 0.001,
        max_epochs: int = 250,
        batch_size: int = 16,
        alpha: float = 0.001,
        dropout: float = 0.10,
        validation_fraction: float = 0.20,
        n_iter_no_change: int = 25,
        tol: float = 1e-4,
        random_state: int = RANDOM_STATE,
    ) -> None:
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.alpha = alpha
        self.dropout = dropout
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.tol = tol
        self.random_state = random_state

    @staticmethod
    def _sigmoid(values: np.ndarray) -> np.ndarray:
        clipped = np.clip(values, -30.0, 30.0)
        return 1.0 / (1.0 + np.exp(-clipped))

    @staticmethod
    def _binary_loss(probabilities: np.ndarray, targets: np.ndarray) -> float:
        probabilities = np.clip(probabilities, 1e-8, 1.0 - 1e-8)
        targets = targets.reshape(-1, 1)
        return float(
            -np.mean(
                targets * np.log(probabilities)
                + (1.0 - targets) * np.log(1.0 - probabilities)
            )
        )

    def _initialise_parameters(
        self, n_features: int, rng: np.random.Generator
    ) -> dict[str, np.ndarray]:
        hidden = self.hidden_dim
        return {
            "W_in": rng.normal(0.0, np.sqrt(2.0 / n_features), (n_features, hidden)),
            "b_in": np.zeros((1, hidden)),
            "W_1": rng.normal(0.0, np.sqrt(2.0 / hidden), (hidden, hidden)),
            "b_1": np.zeros((1, hidden)),
            "W_2": rng.normal(0.0, np.sqrt(2.0 / hidden), (hidden, hidden)),
            "b_2": np.zeros((1, hidden)),
            "W_out": rng.normal(0.0, np.sqrt(1.0 / hidden), (hidden, 1)),
            "b_out": np.zeros((1, 1)),
        }

    def _forward(
        self,
        X: np.ndarray,
        training: bool,
        rng: np.random.Generator | None = None,
    ) -> tuple[np.ndarray, dict[str, np.ndarray | None]]:
        parameters = self.parameters_
        z_in = X @ parameters["W_in"] + parameters["b_in"]
        h_in = np.maximum(z_in, 0.0)
        z_1 = h_in @ parameters["W_1"] + parameters["b_1"]
        h_1 = np.maximum(z_1, 0.0)

        dropout_mask = None
        if training and self.dropout > 0:
            if rng is None:
                raise ValueError("A random generator is required during training.")
            dropout_mask = (
                rng.random(h_1.shape) >= self.dropout
            ).astype(np.float64) / (1.0 - self.dropout)
            h_1_drop = h_1 * dropout_mask
        else:
            h_1_drop = h_1

        z_2 = h_1_drop @ parameters["W_2"] + parameters["b_2"]
        residual_pre_activation = h_in + z_2
        hidden_output = np.maximum(residual_pre_activation, 0.0)
        logits = hidden_output @ parameters["W_out"] + parameters["b_out"]
        probabilities = self._sigmoid(logits)
        cache: dict[str, np.ndarray | None] = {
            "X": X,
            "z_in": z_in,
            "h_in": h_in,
            "z_1": z_1,
            "h_1_drop": h_1_drop,
            "dropout_mask": dropout_mask,
            "residual_pre_activation": residual_pre_activation,
            "hidden_output": hidden_output,
            "probabilities": probabilities,
        }
        return probabilities, cache

    def _gradients(
        self, cache: dict[str, np.ndarray | None], targets: np.ndarray
    ) -> dict[str, np.ndarray]:
        parameters = self.parameters_
        X = np.asarray(cache["X"])
        z_in = np.asarray(cache["z_in"])
        h_in = np.asarray(cache["h_in"])
        z_1 = np.asarray(cache["z_1"])
        h_1_drop = np.asarray(cache["h_1_drop"])
        residual_pre_activation = np.asarray(cache["residual_pre_activation"])
        hidden_output = np.asarray(cache["hidden_output"])
        probabilities = np.asarray(cache["probabilities"])
        dropout_mask = cache["dropout_mask"]

        targets_column = targets.reshape(-1, 1)
        batch_size = max(1, len(targets))
        d_logits = (probabilities - targets_column) / batch_size

        gradients: dict[str, np.ndarray] = {}
        gradients["W_out"] = (
            hidden_output.T @ d_logits + self.alpha * parameters["W_out"]
        )
        gradients["b_out"] = d_logits.sum(axis=0, keepdims=True)

        d_hidden_output = d_logits @ parameters["W_out"].T
        d_residual = d_hidden_output * (residual_pre_activation > 0)
        gradients["W_2"] = h_1_drop.T @ d_residual + self.alpha * parameters["W_2"]
        gradients["b_2"] = d_residual.sum(axis=0, keepdims=True)

        d_h_1 = d_residual @ parameters["W_2"].T
        if dropout_mask is not None:
            d_h_1 = d_h_1 * np.asarray(dropout_mask)
        d_z_1 = d_h_1 * (z_1 > 0)
        gradients["W_1"] = h_in.T @ d_z_1 + self.alpha * parameters["W_1"]
        gradients["b_1"] = d_z_1.sum(axis=0, keepdims=True)

        d_h_in = d_residual + d_z_1 @ parameters["W_1"].T
        d_z_in = d_h_in * (z_in > 0)
        gradients["W_in"] = X.T @ d_z_in + self.alpha * parameters["W_in"]
        gradients["b_in"] = d_z_in.sum(axis=0, keepdims=True)
        return gradients

    def _adam_update(
        self,
        gradients: dict[str, np.ndarray],
        first_moment: dict[str, np.ndarray],
        second_moment: dict[str, np.ndarray],
        step: int,
    ) -> None:
        beta_1 = 0.9
        beta_2 = 0.999
        epsilon = 1e-8
        for name, gradient in gradients.items():
            first_moment[name] = beta_1 * first_moment[name] + (1.0 - beta_1) * gradient
            second_moment[name] = beta_2 * second_moment[name] + (1.0 - beta_2) * (
                gradient**2
            )
            first_corrected = first_moment[name] / (1.0 - beta_1**step)
            second_corrected = second_moment[name] / (1.0 - beta_2**step)
            self.parameters_[name] -= self.learning_rate * first_corrected / (
                np.sqrt(second_corrected) + epsilon
            )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TabularResidualClassifier":
        X_checked, y_checked = check_X_y(X, y, dtype=np.float64, accept_sparse=False)
        unique = np.unique(y_checked)
        if not np.array_equal(unique, np.array([0, 1])):
            raise ValueError("TabularResidualClassifier requires binary labels encoded as 0 and 1.")
        if self.hidden_dim < 2:
            raise ValueError("hidden_dim must be at least 2.")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")

        self.classes_ = np.array([0, 1])
        self.n_features_in_ = X_checked.shape[1]
        rng = np.random.default_rng(self.random_state)

        can_validate = (
            self.validation_fraction > 0
            and np.min(np.bincount(y_checked.astype(int))) >= 2
        )
        if can_validate:
            X_train, X_validation, y_train, y_validation = train_test_split(
                X_checked,
                y_checked.astype(int),
                test_size=self.validation_fraction,
                random_state=self.random_state,
                stratify=y_checked,
            )
        else:
            X_train, y_train = X_checked, y_checked.astype(int)
            X_validation = y_validation = None

        self.parameters_ = self._initialise_parameters(X_checked.shape[1], rng)
        first_moment = {
            name: np.zeros_like(value) for name, value in self.parameters_.items()
        }
        second_moment = {
            name: np.zeros_like(value) for name, value in self.parameters_.items()
        }
        self.loss_curve_: list[float] = []
        self.validation_loss_curve_: list[float] = []
        best_loss = np.inf
        best_parameters = {name: value.copy() for name, value in self.parameters_.items()}
        epochs_without_improvement = 0
        update_step = 0

        for epoch in range(self.max_epochs):
            order = rng.permutation(len(X_train))
            for start in range(0, len(order), self.batch_size):
                batch_index = order[start : start + self.batch_size]
                _, cache = self._forward(X_train[batch_index], training=True, rng=rng)
                gradients = self._gradients(cache, y_train[batch_index])
                update_step += 1
                self._adam_update(
                    gradients, first_moment, second_moment, update_step
                )

            train_probability, _ = self._forward(X_train, training=False)
            train_loss = self._binary_loss(train_probability, y_train)
            self.loss_curve_.append(train_loss)

            if X_validation is None or y_validation is None:
                monitored_loss = train_loss
            else:
                validation_probability, _ = self._forward(X_validation, training=False)
                monitored_loss = self._binary_loss(validation_probability, y_validation)
                self.validation_loss_curve_.append(monitored_loss)

            if monitored_loss < best_loss - self.tol:
                best_loss = monitored_loss
                best_parameters = {
                    name: value.copy() for name, value in self.parameters_.items()
                }
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= self.n_iter_no_change:
                    self.n_iter_ = epoch + 1
                    break
        else:
            self.n_iter_ = self.max_epochs

        self.parameters_ = best_parameters
        self.best_validation_loss_ = float(best_loss)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, ["parameters_", "classes_"])
        X_checked = check_array(X, dtype=np.float64, accept_sparse=False)
        probability, _ = self._forward(X_checked, training=False)
        positive = probability.ravel()
        return np.column_stack([1.0 - positive, positive])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    family: str
    substantive: bool
    estimator: BaseEstimator
    description: str


def build_model_specs() -> list[ModelSpec]:
    """Return the fixed Q3.1-Q3.2 model set."""
    return [
        ModelSpec(
            key="majority_baseline",
            label="Majority-class baseline",
            family="naive baseline",
            substantive=False,
            estimator=DummyClassifier(strategy="most_frequent", random_state=RANDOM_STATE),
            description="Ignores all predictors and always returns the training-fold majority class.",
        ),
        ModelSpec(
            key="logistic_regression",
            label="Logistic Regression",
            family="traditional",
            substantive=True,
            estimator=LogisticRegression(
                C=1.0,
                max_iter=2_000,
                solver="liblinear",
                random_state=RANDOM_STATE,
            ),
            description="Regularised linear probability boundary for a transparent tabular comparator.",
        ),
        ModelSpec(
            key="random_forest",
            label="Random Forest",
            family="traditional",
            substantive=True,
            estimator=RandomForestClassifier(
                n_estimators=300,
                min_samples_leaf=2,
                max_features="sqrt",
                n_jobs=1,
                random_state=RANDOM_STATE,
            ),
            description="Non-linear tree ensemble that can capture interactions without feature linearity.",
        ),
        ModelSpec(
            key="mlp",
            label="Multilayer Perceptron",
            family="neural",
            substantive=True,
            estimator=MLPClassifier(
                hidden_layer_sizes=(32, 16),
                activation="relu",
                solver="adam",
                alpha=0.001,
                batch_size=16,
                learning_rate_init=0.001,
                max_iter=500,
                early_stopping=True,
                validation_fraction=0.20,
                n_iter_no_change=30,
                random_state=RANDOM_STATE,
            ),
            description="Required feed-forward MLP with two hidden layers and early stopping.",
        ),
        ModelSpec(
            key="residual_tabular_nn",
            label="Residual Tabular Neural Network",
            family="neural",
            substantive=True,
            estimator=TabularResidualClassifier(random_state=RANDOM_STATE),
            description="Task-appropriate dense residual network for the engineered tabular features.",
        ),
    ]


def build_model_pipeline(spec: ModelSpec) -> ImbalancedPipeline:
    """Place Q2.4 preparation and a model in one fold-local pipeline."""
    steps: list[tuple[str, object]] = [
        ("feature_engineering", ReliabilityFeatureEngineer()),
        ("preprocess", build_column_preprocessor()),
    ]
    if spec.substantive:
        steps.append(("balance", RandomOverSampler(random_state=RANDOM_STATE)))
    steps.append(("model", spec.estimator))
    return ImbalancedPipeline(steps=steps)

