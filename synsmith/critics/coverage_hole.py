"""Coverage Hole Finder: density-ratio coverage signal (vanilla GAN derivation).

A trained vanilla GAN discriminator implicitly estimates the density ratio
``p_real(x) / (p_real(x) + p_synth(x))``. We construct that estimator
explicitly with a logistic regression on TF-IDF features: train it to
classify real vs synthetic, then for each *real* sample compute the
predicted probability of being real. Real samples the classifier most
confidently calls real are the modes of the real distribution that the
synthetic distribution has not covered.

The top-K most-uncovered real exemplars become *few-shot hints* in the
next generator prompt: "Here are real examples whose style and content
the current synthetic batch is failing to reproduce. Generate more like
these."

This converts a fuzzy "missing modes" feedback into a concrete exemplar
set that the prompt updater can ground its rewrite in.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pydantic import BaseModel, Field
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from synsmith.schema import RealExample, SyntheticSample


@dataclass
class CoverageHoleConfig:
    top_k: int = 5
    min_real: int = 5
    min_synth: int = 5
    ngram_range: tuple[int, int] = (1, 2)
    stratify_by_label: bool = True
    """Fix F6: when True, the top-K hole selection is stratified across
    real-sample labels so every class gets at least one exemplar in the
    output when its class has any uncovered real samples. Otherwise the
    global top-K can be dominated by one or two classes and the under-
    represented classes never surface coverage-hole exemplars to the
    updater, perpetuating per-class undercoverage. Default-on; set False
    for backward-compat with the v2.9.x ranking."""


class CoverageHole(BaseModel):
    text: str
    label: str | None = None
    p_real: float = Field(..., description="Classifier's probability that this is real.")


class CoverageHoleResult(BaseModel):
    holes: list[CoverageHole] = Field(default_factory=list)
    classifier_auroc: float = 0.5
    notes: str = ""


class CoverageHoleFinder:
    """Density-ratio coverage analysis with a tiny LR classifier."""

    def __init__(self, config: CoverageHoleConfig | None = None) -> None:
        self.config = config or CoverageHoleConfig()

    def find(
        self,
        real: list[RealExample],
        synthetic: list[SyntheticSample],
    ) -> CoverageHoleResult:
        if len(real) < self.config.min_real or len(synthetic) < self.config.min_synth:
            return CoverageHoleResult(
                holes=[], classifier_auroc=0.5, notes="not enough samples to fit"
            )

        real_texts = [r.text for r in real]
        synth_texts = [s.text for s in synthetic]
        all_texts = real_texts + synth_texts
        y = np.concatenate(
            [np.ones(len(real_texts)), np.zeros(len(synth_texts))]
        )

        vec = TfidfVectorizer(ngram_range=self.config.ngram_range, min_df=1)
        X = vec.fit_transform(all_texts)

        try:
            clf = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced")
            clf.fit(X, y)
        except Exception as exc:
            return CoverageHoleResult(holes=[], classifier_auroc=0.5, notes=str(exc))

        p_real_all = clf.predict_proba(X)[:, 1]
        real_p = p_real_all[: len(real_texts)]

        # AUROC ~ how well the classifier separates real from synthetic.
        auroc = self._auroc(y, p_real_all)

        if self.config.stratify_by_label:
            # Fix F6: round-robin pick across labels, taking the highest
            # p_real per class first, then the second-highest per class,
            # until top_k is filled. Classes with no real examples are
            # skipped. Classes with fewer examples than the round count
            # contribute fewer picks (no padding).
            order_by_label: dict[str, list[int]] = {}
            for idx in np.argsort(-real_p):
                lbl = real[int(idx)].label or "_"
                order_by_label.setdefault(lbl, []).append(int(idx))
            labels_sorted = sorted(order_by_label.keys())
            picked: list[int] = []
            seen: set[int] = set()
            round_n = 0
            while len(picked) < self.config.top_k:
                progress = False
                for lbl in labels_sorted:
                    pool = order_by_label[lbl]
                    if round_n < len(pool):
                        idx = pool[round_n]
                        if idx not in seen:
                            picked.append(idx)
                            seen.add(idx)
                            progress = True
                            if len(picked) >= self.config.top_k:
                                break
                if not progress:
                    break
                round_n += 1
            order = picked
        else:
            order = list(np.argsort(-real_p)[: self.config.top_k])
        holes = [
            CoverageHole(
                text=real[int(i)].text,
                label=real[int(i)].label,
                p_real=float(real_p[int(i)]),
            )
            for i in order
        ]
        return CoverageHoleResult(holes=holes, classifier_auroc=auroc)

    @staticmethod
    def _auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
        """Pairwise AUROC; ties broken at 0.5. Avoids importing sklearn.metrics."""
        pos = y_score[y_true == 1]
        neg = y_score[y_true == 0]
        if len(pos) == 0 or len(neg) == 0:
            return 0.5
        wins = (pos[:, None] > neg[None, :]).sum()
        ties = (pos[:, None] == neg[None, :]).sum()
        total = len(pos) * len(neg)
        return float((wins + 0.5 * ties) / total)

    def render_for_prompt(self, result: CoverageHoleResult) -> str:
        if not result.holes:
            return ""
        bullets = "\n".join(
            f"- (label={h.label or '?'}, p_real={h.p_real:.2f}) {h.text}"
            for h in result.holes
        )
        return (
            "The synthetic distribution is failing to cover real examples "
            "like these. The next batch should produce more in this style:\n"
            + bullets
        )
