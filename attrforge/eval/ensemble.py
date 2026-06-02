"""Cross-condition classifier ensembling.

The paper's main empirical contribution (Section 7.3): a downstream
classifier trained on synthetic-batch (condition A) and another trained
on synthetic-batch (condition B) often learn complementary decision
boundaries on the held-out real test set. Logit-averaging the two
classifiers extracts the shared signal and softens the boundary where
they disagree, which reduces seed variance and lifts worst-class F1.

Public API:

    from attrforge.eval import (
        CrossConditionEnsemble,
        EnsembleResult,
        ensemble_pair,
        ensemble_set,
    )

    ens = CrossConditionEnsemble(
        backend="sentence-transformer",   # or "tfidf"
        classifier="logistic-regression", # default
    )
    result = ens.fit_evaluate(
        condition_batches={
            "self_critique": [SyntheticSample, ...],
            "full_attrforge": [SyntheticSample, ...],
        },
        real_train=[RealExample, ...],
        real_test=[RealExample, ...],
        seed=17,
    )
    print(result.macro_f1, result.worst_class_f1)
    # 0.947  0.833

The ensemble.proba field carries each condition's predicted log-probability
matrix per held-out test item, so pairs / all-iter ensembles can be re-
aggregated post hoc without rerunning the classifier fits.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Iterable, Sequence

import numpy as np
from pydantic import BaseModel, Field
from sklearn.linear_model import LogisticRegression

from attrforge.schema import RealExample, SyntheticSample


@dataclass
class CrossConditionEnsembleConfig:
    """Configuration for the cross-condition ensemble.

    backend: "sentence-transformer" (recommended) or "tfidf".
    classifier: "logistic-regression" (only option for now).
    C: LR regularization strength.
    seed: RNG seed for the classifier.
    """

    backend: str = "sentence-transformer"
    classifier: str = "logistic-regression"
    C: float = 1.0
    seed: int = 17
    sentence_transformer_model: str = "all-MiniLM-L6-v2"
    tfidf_ngram_range: tuple[int, int] = (1, 2)
    tfidf_min_df: int = 1


class EnsembleResult(BaseModel):
    """Result of evaluating one ensemble (either a pair or an all-iter set).

    macro_f1, worst_class_f1: aggregate metrics on the held-out test.
    per_class_f1, confusion: per-class breakdown.
    pair_or_set: tuple of condition names that were ensembled.
    n_test: held-out test size.
    label_set: list of class labels in alphabetical order.
    """

    macro_f1: float
    worst_class_f1: float
    accuracy: float
    per_class_f1: dict[str, float] = Field(default_factory=dict)
    confusion: dict[str, dict[str, int]] = Field(default_factory=dict)
    pair_or_set: tuple[str, ...] = Field(default_factory=tuple)
    n_test: int = 0
    label_set: list[str] = Field(default_factory=list)


def _macro_and_worst(y_true: np.ndarray, y_pred: np.ndarray, labels: Sequence[str]) -> tuple[float, float, dict[str, float], dict[str, dict[str, int]]]:
    per_f1: dict[str, float] = {}
    cm: dict[str, dict[str, int]] = {l: {l2: 0 for l2 in labels} for l in labels}
    for t, p in zip(y_true, y_pred):
        cm[t][p] += 1
    for lbl in labels:
        tp = cm[lbl][lbl]
        fp = sum(cm[l][lbl] for l in labels if l != lbl)
        fn = sum(cm[lbl][l] for l in labels if l != lbl)
        if tp + fp == 0 or tp + fn == 0:
            per_f1[lbl] = 0.0
            continue
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        per_f1[lbl] = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    macro = float(np.mean(list(per_f1.values())))
    worst = float(min(per_f1.values())) if per_f1 else 0.0
    return macro, worst, per_f1, cm


def _featurize(
    texts: Iterable[str], cfg: CrossConditionEnsembleConfig, encoder=None
) -> tuple[np.ndarray, object | None]:
    """Featurize a list of texts under the configured backend.

    Returns (feature_matrix, fitted_encoder_or_None). For sentence-transformer
    the encoder is the loaded model; for tfidf it is the fitted TfidfVectorizer.
    """
    texts = list(texts)
    if cfg.backend == "sentence-transformer":
        if encoder is None:
            from sentence_transformers import SentenceTransformer  # noqa: WPS433

            encoder = SentenceTransformer(cfg.sentence_transformer_model)
        X = encoder.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return X, encoder
    if cfg.backend == "tfidf":
        from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: WPS433

        if encoder is None:
            encoder = TfidfVectorizer(
                ngram_range=cfg.tfidf_ngram_range, min_df=cfg.tfidf_min_df
            ).fit(texts)
        X = encoder.transform(texts).toarray()
        return X, encoder
    raise ValueError(f"unknown backend: {cfg.backend!r}")


def _featurize_test(texts: Iterable[str], cfg: CrossConditionEnsembleConfig, encoder) -> np.ndarray:
    texts = list(texts)
    if cfg.backend == "sentence-transformer":
        return encoder.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    if cfg.backend == "tfidf":
        return encoder.transform(texts).toarray()
    raise ValueError(f"unknown backend: {cfg.backend!r}")


class CrossConditionEnsemble:
    """Train a classifier per condition, then ensemble via logit averaging.

    Workflow:
        1. fit_per_condition(condition_batches, real_train, real_test) trains
           one classifier per condition on real_train ∪ synthetic-batch(c)
           and caches the per-test predicted-probability matrix.
        2. ensemble_pair(c_a, c_b) / ensemble_set([c1, c2, ...]) computes the
           logit-averaged ensemble on the cached probas.
    """

    def __init__(self, config: CrossConditionEnsembleConfig | None = None):
        self.config = config or CrossConditionEnsembleConfig()
        self._proba: dict[str, np.ndarray] = {}
        self._labels: list[str] = []
        self._y_test: np.ndarray | None = None
        self._encoder = None

    def fit_per_condition(
        self,
        condition_batches: dict[str, Sequence[SyntheticSample]],
        real_train: Sequence[RealExample],
        real_test: Sequence[RealExample],
    ) -> "CrossConditionEnsemble":
        cfg = self.config
        real_train_texts = [r.text for r in real_train]
        real_train_labels = np.asarray([r.label for r in real_train])
        test_texts = [r.text for r in real_test]
        y_test = np.asarray([r.label for r in real_test])
        labels = sorted(set(y_test.tolist()))

        # Featurize the real train texts once. For sentence-transformer the
        # encoder is constructed inside _featurize; for tfidf we have to
        # refit per condition because the synthetic-text vocabulary differs.
        X_real_train, encoder = _featurize(real_train_texts, cfg)
        X_test = _featurize_test(test_texts, cfg, encoder)

        for cond, samples in condition_batches.items():
            synth_texts = [s.text for s in samples]
            synth_labels = np.asarray(
                [s.requested_attributes.get("intent", "?") for s in samples]
            )
            if cfg.backend == "sentence-transformer":
                X_synth = _featurize_test(synth_texts, cfg, encoder)
                X_tr = np.concatenate([X_real_train, X_synth], axis=0)
            else:
                # For TF-IDF we refit on the union vocabulary per condition
                # to maximize coverage.
                X_tr, encoder_cond = _featurize(
                    real_train_texts + synth_texts, cfg
                )
                X_test = _featurize_test(test_texts, cfg, encoder_cond)
            y_tr = np.concatenate([real_train_labels, synth_labels])
            clf = LogisticRegression(
                max_iter=2000,
                C=cfg.C,
                class_weight="balanced",
                random_state=cfg.seed,
            )
            clf.fit(X_tr, y_tr)
            proba = clf.predict_proba(X_test)
            col_idx = [list(clf.classes_).index(lbl) for lbl in labels]
            self._proba[cond] = proba[:, col_idx]

        self._labels = labels
        self._y_test = y_test
        self._encoder = encoder
        return self

    def ensemble_pair(self, c_a: str, c_b: str) -> EnsembleResult:
        """Logit-average two conditions' classifiers and evaluate."""
        if c_a not in self._proba or c_b not in self._proba:
            raise KeyError(f"conditions not fit: {c_a!r} or {c_b!r}")
        ens = (self._proba[c_a] + self._proba[c_b]) / 2.0
        y_pred = np.asarray([self._labels[i] for i in ens.argmax(axis=1)])
        macro, worst, per_f1, cm = _macro_and_worst(
            self._y_test, y_pred, self._labels
        )
        acc = float((y_pred == self._y_test).mean())
        return EnsembleResult(
            macro_f1=macro,
            worst_class_f1=worst,
            accuracy=acc,
            per_class_f1=per_f1,
            confusion=cm,
            pair_or_set=(c_a, c_b),
            n_test=len(self._y_test),
            label_set=self._labels,
        )

    def ensemble_set(self, conditions: Sequence[str]) -> EnsembleResult:
        """Logit-average N conditions' classifiers and evaluate."""
        missing = [c for c in conditions if c not in self._proba]
        if missing:
            raise KeyError(f"conditions not fit: {missing}")
        ens = np.mean(
            np.stack([self._proba[c] for c in conditions], axis=0), axis=0
        )
        y_pred = np.asarray([self._labels[i] for i in ens.argmax(axis=1)])
        macro, worst, per_f1, cm = _macro_and_worst(
            self._y_test, y_pred, self._labels
        )
        acc = float((y_pred == self._y_test).mean())
        return EnsembleResult(
            macro_f1=macro,
            worst_class_f1=worst,
            accuracy=acc,
            per_class_f1=per_f1,
            confusion=cm,
            pair_or_set=tuple(conditions),
            n_test=len(self._y_test),
            label_set=self._labels,
        )

    def best_pair(self) -> tuple[tuple[str, str], EnsembleResult]:
        """Return the (pair, result) with highest macro F1 across all pairs."""
        conds = sorted(self._proba.keys())
        best: tuple[tuple[str, str], EnsembleResult] | None = None
        for a, b in combinations(conds, 2):
            r = self.ensemble_pair(a, b)
            if best is None or r.macro_f1 > best[1].macro_f1:
                best = ((a, b), r)
        if best is None:
            raise RuntimeError("no pairs evaluated; at least 2 conditions needed")
        return best


def ensemble_pair(
    condition_batches: dict[str, Sequence[SyntheticSample]],
    real_train: Sequence[RealExample],
    real_test: Sequence[RealExample],
    pair: tuple[str, str],
    config: CrossConditionEnsembleConfig | None = None,
) -> EnsembleResult:
    """Convenience: fit and ensemble a single pair in one call."""
    ens = CrossConditionEnsemble(config).fit_per_condition(
        condition_batches, real_train, real_test
    )
    return ens.ensemble_pair(*pair)


def ensemble_set(
    condition_batches: dict[str, Sequence[SyntheticSample]],
    real_train: Sequence[RealExample],
    real_test: Sequence[RealExample],
    conditions: Sequence[str],
    config: CrossConditionEnsembleConfig | None = None,
) -> EnsembleResult:
    """Convenience: fit and ensemble a list of conditions in one call."""
    ens = CrossConditionEnsemble(config).fit_per_condition(
        condition_batches, real_train, real_test
    )
    return ens.ensemble_set(conditions)
