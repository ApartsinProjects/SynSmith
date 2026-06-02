"""Diversity Auditor.

Combines two signals:

1. **Deterministic coverage**: per-attribute fraction of allowed values
   observed, plus an embedding (or TF-IDF fallback) near-duplicate rate.
   These are cheap, reproducible numbers the loop can plot directly.

2. **LLM judgment**: a structured natural-language audit that names
   missing and overrepresented modes and recommends concrete additions
   to the generator prompt.

We always run (1); (2) is optional and skipped when no client is given.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import yaml

from attrforge.llm import LLMClient, json_chat
from attrforge.prompts import AUDITOR_SYSTEM, AUDITOR_USER_TEMPLATE
from attrforge.schema import AttributeSchema, DiversityReport, SyntheticSample


@dataclass
class AuditorConfig:
    near_duplicate_threshold: float = 0.92
    max_samples_in_prompt: int = 30
    use_embeddings: bool = False
    embedding_model: str = "all-MiniLM-L6-v2"


class DiversityAuditor:
    def __init__(
        self,
        schema: AttributeSchema,
        client: LLMClient | None = None,
        config: AuditorConfig | None = None,
    ) -> None:
        self.schema = schema
        self.client = client
        self.config = config or AuditorConfig()
        self._embedder = None

    def audit(self, batch: list[SyntheticSample]) -> DiversityReport:
        if not batch:
            return DiversityReport(summary="empty batch")

        coverage = self._coverage(batch)
        ndr = self._near_duplicate_rate(batch)

        if self.client is None:
            return DiversityReport(
                summary="deterministic-only audit (no LLM client)",
                near_duplicate_rate=ndr,
                coverage=coverage,
                recommendations=self._mechanical_recommendations(coverage),
            )

        return self._llm_audit(batch, coverage, ndr)

    def _coverage(self, batch: list[SyntheticSample]) -> dict[str, float]:
        """Fraction of allowed values per attribute that show up in the batch."""
        coverage: dict[str, float] = {}
        for name, allowed in self.schema.attributes.items():
            seen = {s.requested_attributes.get(name) for s in batch}
            seen.discard(None)
            coverage[name] = len(seen) / max(1, len(allowed))
        return coverage

    def _near_duplicate_rate(self, batch: list[SyntheticSample]) -> float:
        """Fraction of samples that have a near-twin within the batch.

        Uses sentence-transformer embeddings when ``use_embeddings`` is on,
        otherwise falls back to TF-IDF cosine. Both produce a symmetric
        similarity matrix; we count any sample whose max non-self similarity
        crosses ``near_duplicate_threshold``.
        """
        texts = [s.text for s in batch]
        if len(texts) < 2:
            return 0.0
        sim = self._similarity_matrix(texts)
        np.fill_diagonal(sim, 0.0)
        max_sim = sim.max(axis=1)
        return float((max_sim >= self.config.near_duplicate_threshold).mean())

    def _similarity_matrix(self, texts: list[str]) -> np.ndarray:
        if self.config.use_embeddings:
            try:
                if self._embedder is None:
                    from sentence_transformers import SentenceTransformer

                    self._embedder = SentenceTransformer(self.config.embedding_model)
                emb = self._embedder.encode(texts, normalize_embeddings=True)
                return emb @ emb.T
            except Exception:
                pass
        from sklearn.feature_extraction.text import TfidfVectorizer

        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1).fit_transform(texts)
        norm = np.sqrt((vec.multiply(vec)).sum(axis=1))
        norm[norm == 0] = 1.0
        return np.asarray((vec @ vec.T).todense()) / (norm @ norm.T)

    def _mechanical_recommendations(self, coverage: dict[str, float]) -> list[str]:
        recs: list[str] = []
        for name, frac in coverage.items():
            if frac < 0.75:
                missing = [
                    v
                    for v in self.schema.values(name)
                    if frac < 1.0
                ][:3]
                if missing:
                    recs.append(
                        f"Increase coverage of attribute '{name}' "
                        f"(values to add: {', '.join(missing)})."
                    )
        return recs

    def _llm_audit(
        self,
        batch: list[SyntheticSample],
        coverage: dict[str, float],
        near_duplicate_rate: float,
    ) -> DiversityReport:
        schema_str = yaml.safe_dump(self.schema.attributes, sort_keys=False)
        sample = batch[: self.config.max_samples_in_prompt]
        # Text first, requested attrs LAST: the user reads the text and
        # judges nuance-level diversity before being primed by the labels.
        batch_block = "\n\n".join(
            f"id: {s.sample_id}\ntext: {s.text[:300]}\nrequested_attrs: {s.requested_attributes}"
            for s in sample
        )
        coverage_block = "\n".join(
            f"  {name}: {frac:.2f}" for name, frac in coverage.items()
        )
        user_msg = AUDITOR_USER_TEMPLATE.format(
            attribute_schema=schema_str,
            batch_block=batch_block,
            coverage_block=coverage_block,
        )
        obj = json_chat(
            self.client,
            AUDITOR_SYSTEM,
            [{"role": "user", "content": user_msg}],
            temperature=0.2,
            max_tokens=800,
            retries=1,
        )
        return DiversityReport(
            summary=str(obj.get("summary", "")),
            missing_modes=[str(x) for x in obj.get("missing_modes", [])],
            overrepresented_modes=[str(x) for x in obj.get("overrepresented_modes", [])],
            near_duplicate_rate=float(
                obj.get("near_duplicate_rate", near_duplicate_rate) or near_duplicate_rate
            ),
            recommendations=[str(x) for x in obj.get("recommendations", [])],
            coverage=coverage,
        )
