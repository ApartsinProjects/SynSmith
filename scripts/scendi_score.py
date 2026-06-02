"""Scendi score: prompt-aware diversity decomposition via Schur complement
of the sentence-transformer Gram matrix.

Jandaghi et al. (arXiv:2412.18645) introduce the Scendi score: a Vendi-style
diversity measure that decomposes diversity into a prompt-driven component
(induced by the conditioning prompt) and an intrinsic-model component
(the model's residual diversity given the prompt). Concretely, given a kernel
matrix K over (prompt, output) pairs and a kernel matrix K_P over the prompts
alone, the Scendi score is:

    Scendi(K, K_P) = Vendi(K - K_P K_P^{-1} K_P^T)

where K - K_P K_P^{-1} K_P^T is the Schur complement that removes the
prompt-induced diversity from the joint kernel.

For PromptForge, the "prompt" is the active critic-stack prompt at the
condition's final iteration; the "output" is the synthetic sample text.
Iteration is supposed to grow PROMPT-DRIVEN diversity (the updater rewrites
the prompt to surface uncovered modes), so we expect:

    iterated cluster Vendi  >  non-iterated cluster Vendi  (already shown)
    iterated cluster Scendi >  non-iterated cluster Scendi  if iteration
        injects intrinsic-model diversity (the seven-critic loop ACTUALLY
        unlocks model capacity that the naive prompt under-elicits)

    iterated cluster Scendi ~~ non-iterated cluster Scendi  if iteration
        is purely a prompt-rewrite effect (the same model would produce
        the same intrinsic diversity under any reasonable prompt)

The decomposition lets the paper state: of the 2x Vendi gain over
non-iterated, X% is prompt-driven and (100-X)% is intrinsic-model.

Outputs:
    experiments/<base>_aggregated/scendi.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import attrforge  # noqa: E402
from attrforge.schema import SyntheticSample, load_jsonl  # noqa: E402


def load_synth(cond_dir: Path) -> list[SyntheticSample]:
    out = []
    for iter_dir in sorted(cond_dir.glob("*/iter_*")):
        sj = iter_dir / "samples.jsonl"
        if sj.exists():
            for r in load_jsonl(sj):
                out.append(SyntheticSample.model_validate(r))
    return out


def vendi_score(emb_matrix: np.ndarray) -> float:
    """Vendi(K) = exp( entropy(eigvals of K/n) ) where K = E @ E.T."""
    K = emb_matrix @ emb_matrix.T
    n = K.shape[0]
    if n == 0:
        return 0.0
    eigvals = np.linalg.eigvalsh(K / n)
    eigvals = eigvals[eigvals > 1e-12]
    if eigvals.size == 0:
        return 0.0
    eigvals = eigvals / eigvals.sum()
    entropy = -(eigvals * np.log(eigvals)).sum()
    return float(np.exp(entropy))


def scendi_score(
    emb_outputs: np.ndarray, emb_prompts: np.ndarray, ridge: float = 1e-3
) -> float:
    """Scendi score: Vendi on the Schur-complemented kernel that removes
    prompt-induced diversity.

    Following Jandaghi et al. (arXiv:2412.18645). Let:
        K_O = E_O @ E_O.T            # output-output kernel
        K_OP = E_O @ E_P.T           # output-prompt cross-kernel
        K_P = E_P @ E_P.T + ridge*I  # prompt-prompt kernel, ridge-regularized

    The Schur-complemented residual kernel is:
        K_res = K_O - K_OP @ K_P^{-1} @ K_OP.T

    Scendi = Vendi(K_res).

    emb_outputs : (n, d) row-normalized embeddings of synthetic outputs
    emb_prompts : (n, d) row-normalized embeddings of each output's prompt
                  (same prompt for every output in a single iteration, so n
                  copies of the iteration prompt)
    """
    n = emb_outputs.shape[0]
    if n == 0:
        return 0.0
    K_O = emb_outputs @ emb_outputs.T
    K_OP = emb_outputs @ emb_prompts.T
    K_P = emb_prompts @ emb_prompts.T + ridge * np.eye(n)
    try:
        K_P_inv = np.linalg.inv(K_P)
    except np.linalg.LinAlgError:
        K_P_inv = np.linalg.pinv(K_P)
    K_res = K_O - K_OP @ K_P_inv @ K_OP.T
    # Symmetrize to clean up any floating-point asymmetry.
    K_res = (K_res + K_res.T) / 2.0
    eigvals = np.linalg.eigvalsh(K_res / n)
    eigvals = eigvals[eigvals > 1e-12]
    if eigvals.size == 0:
        return 0.0
    eigvals = eigvals / eigvals.sum()
    entropy = -(eigvals * np.log(eigvals)).sum()
    return float(np.exp(entropy))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    args = ap.parse_args()

    import os
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    from sentence_transformers import SentenceTransformer

    # Force CPU encoding so we don't fight the R1 background run for the GPU.
    enc = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

    seed_dirs = sorted((REPO / "experiments").glob(f"{args.base}_seed*"))
    if not seed_dirs:
        print(f"No seed dirs matching {args.base}_seed*")
        return

    bag: dict[str, list[dict[str, float]]] = defaultdict(list)

    for sd in seed_dirs:
        try:
            seed = int(sd.name.split("seed")[-1])
        except ValueError:
            continue
        for cond_dir in sorted(p for p in sd.iterdir() if p.is_dir() and p.name not in {"audit", "aggregated"}):
            samples = load_synth(cond_dir)
            if not samples:
                continue
            texts = [s.text for s in samples]
            emb_outputs = enc.encode(texts, normalize_embeddings=True, show_progress_bar=False)
            # Reconstruct the prompt per-iteration. For naive / few_shot,
            # iterations=1, prompt is the initial config prompt. For iterated
            # conditions, we collect each iteration's prompt.txt.
            prompts_per_iter = {}
            for iter_dir in sorted(cond_dir.glob("*/iter_*")):
                ptxt = iter_dir / "prompt.txt"
                if ptxt.exists():
                    prompts_per_iter[iter_dir.name] = ptxt.read_text(encoding="utf-8")
            # Map each sample to the iteration's prompt by finding the
            # iter_NNN dir that contains its samples.jsonl.
            sample_prompts: list[str] = []
            for iter_dir in sorted(cond_dir.glob("*/iter_*")):
                sj = iter_dir / "samples.jsonl"
                if not sj.exists():
                    continue
                rows = list(load_jsonl(sj))
                pr = prompts_per_iter.get(iter_dir.name, "")
                for _ in rows:
                    sample_prompts.append(pr)
            # If prompt reconstruction failed, fall back to a constant
            # "no-prompt" so Scendi degenerates to Vendi (informative but
            # not the prompt-decomposed view).
            if len(sample_prompts) != len(texts) or not any(sample_prompts):
                sample_prompts = ["[no prompt logged]"] * len(texts)
            emb_prompts = enc.encode(sample_prompts, normalize_embeddings=True, show_progress_bar=False)

            v = vendi_score(emb_outputs)
            s = scendi_score(emb_outputs, emb_prompts)
            bag[cond_dir.name].append({"seed": seed, "vendi": v, "scendi": s, "prompt_residual": v - s})

    conds = [
        "naive", "few_shot", "self_critique", "realism_only",
        "diversity_only", "full_classic", "full_attrforge",
        "no_pack", "no_mode_seeking", "no_mode_hunter", "no_coverage_hole",
    ]
    print()
    print(f"{'condition':<20} {'Vendi':<22} {'Scendi':<22} {'V - S (prompt)':<22}")
    for c in conds:
        rows = bag.get(c, [])
        if not rows:
            continue
        def fmt(key: str) -> str:
            v = [r[key] for r in rows]
            return f"{statistics.mean(v):.3f} +- {statistics.stdev(v) if len(v) > 1 else 0:.3f}"
        print(f"{c:<20} {fmt('vendi'):<22} {fmt('scendi'):<22} {fmt('prompt_residual'):<22}")

    out_dir = REPO / "experiments" / f"{args.base}_aggregated"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {c: bag[c] for c in conds if c in bag}
    (out_dir / "scendi.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved: {out_dir}/scendi.json")

    # Compare iterated vs non-iterated cluster on Scendi for the paper claim.
    non_iter = [statistics.mean(r["scendi"] for r in bag[c]) for c in ["naive", "few_shot"] if c in bag]
    iter_set = [statistics.mean(r["scendi"] for r in bag[c]) for c in ["self_critique", "realism_only", "diversity_only", "full_classic", "full_attrforge"] if c in bag]
    if non_iter and iter_set:
        print()
        print(f"Non-iterated Scendi mean: {statistics.mean(non_iter):.3f}")
        print(f"Iterated     Scendi mean: {statistics.mean(iter_set):.3f}")
        ratio = statistics.mean(iter_set) / max(statistics.mean(non_iter), 1e-9)
        print(f"Iterated/non-iterated Scendi ratio: {ratio:.2f}x")

        # Same for Vendi for paper reporting (already in reaudit_fixed but recomputed here for sanity).
        non_iter_v = [statistics.mean(r["vendi"] for r in bag[c]) for c in ["naive", "few_shot"] if c in bag]
        iter_set_v = [statistics.mean(r["vendi"] for r in bag[c]) for c in ["self_critique", "realism_only", "diversity_only", "full_classic", "full_attrforge"] if c in bag]
        ratio_v = statistics.mean(iter_set_v) / max(statistics.mean(non_iter_v), 1e-9)
        print(f"Iterated/non-iterated Vendi  ratio: {ratio_v:.2f}x")
        # Prompt-driven vs intrinsic decomposition:
        # gain_total = iter_vendi - non_iter_vendi
        # gain_intrinsic ~ iter_scendi - non_iter_scendi
        # gain_prompt = gain_total - gain_intrinsic
        gain_total = statistics.mean(iter_set_v) - statistics.mean(non_iter_v)
        gain_intrinsic = statistics.mean(iter_set) - statistics.mean(non_iter)
        gain_prompt = gain_total - gain_intrinsic
        if abs(gain_total) > 1e-9:
            pct_intrinsic = 100.0 * gain_intrinsic / gain_total
            pct_prompt = 100.0 * gain_prompt / gain_total
            print(f"Decomposition of iterated-vs-non-iterated Vendi gain ({gain_total:+.2f}):")
            print(f"  intrinsic-model share: {pct_intrinsic:+.1f}% ({gain_intrinsic:+.2f})")
            print(f"  prompt-driven share:   {pct_prompt:+.1f}% ({gain_prompt:+.2f})")


if __name__ == "__main__":
    main()
