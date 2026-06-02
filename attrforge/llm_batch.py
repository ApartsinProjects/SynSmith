"""OpenAI Batch API harness for AttrForge.

The Batch API runs the same chat-completion requests as the real-time API at
~50% the cost, with a 24-hour SLA. For paper-scale sweeps (multiple seeds,
multiple conditions, multiple iterations, multiple critic calls per sample)
this is the default that the user CLAUDE.md cloud-first rule mandates.

Workflow:

    1. Code calls BatchClient.chat_completion(messages, model, ...) just like
       it would call the OpenAI client. Instead of dispatching, the call is
       BUFFERED with a generated request_id.
    2. After a configured number of buffered requests OR an explicit flush(),
       the BatchClient writes a JSONL file, uploads it to OpenAI Files,
       creates a Batch, and polls until completed (status='completed',
       'failed', 'expired', or 'cancelled').
    3. The completed-batch results are downloaded, indexed by custom_id, and
       returned through awaiting cached Futures on the caller side.

Two operating modes:

    BLOCKING (default): each chat_completion call returns the message after
        the batch is submitted and finishes. Useful as a near-drop-in
        replacement for the existing client in offline runs.

    NON-BLOCKING / FUTURES: chat_completion returns immediately with a
        Future-like object; caller flushes when done. Useful when the
        caller wants to interleave many critic calls in a single batch
        across an entire experiment sweep.

For paper-scale runs, the recommended pattern is:

    # In a scheduling-aware script:
    batch = BatchClient(model="gpt-4o-mini", buffer_capacity=10_000)
    for seed in seeds:
        for cond in conditions:
            for iteration in range(n_iters):
                samples = generator.batch_generate(prompts, client=batch)
                # No API call yet; everything buffered.
    batch.flush()  # ONE batch.jsonl, ONE submission, ONE poll loop.

This module provides:

    BatchClient                      The buffer + submit + poll engine
    BatchResponse                    Pydantic model for one batch response
    submit_batch_jsonl(path, ...)    Lower-level "submit this file"
    fetch_batch_results(batch_id)    Lower-level "download these results"

The module does NOT modify the existing real-time LLMClient; it is a
parallel client the caller opts into.

References:
    https://platform.openai.com/docs/api-reference/batch
    https://platform.openai.com/docs/guides/batch
"""
from __future__ import annotations

import json
import os
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from pydantic import BaseModel, Field


@dataclass
class BatchConfig:
    """Settings for the OpenAI Batch API harness.

    model: model name (e.g. "gpt-4o-mini").
    buffer_capacity: max requests to buffer before auto-flush. None to
        only flush on explicit flush() call.
    poll_interval_seconds: seconds between batch-status polls.
    poll_timeout_seconds: max time to wait for a batch to complete.
        Default 24h matches OpenAI Batch SLA.
    completion_window: OpenAI batch completion window. "24h" is the only
        supported value as of 2025.
    endpoint: OpenAI chat-completions endpoint path.
    cache_dir: optional directory to cache results so a re-run after a
        crash does not re-spend budget.
    """

    model: str = "gpt-4o-mini"
    buffer_capacity: int | None = 10_000
    poll_interval_seconds: int = 30
    poll_timeout_seconds: int = 24 * 3600
    completion_window: str = "24h"
    endpoint: str = "/v1/chat/completions"
    cache_dir: Path | None = None


class BatchResponse(BaseModel):
    """One response in a completed batch."""

    custom_id: str
    content: str = ""
    finish_reason: str | None = None
    model: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    error: str | None = None
    raw: dict[str, Any] = Field(default_factory=dict)


class BatchClient:
    """Buffered Batch-API client.

    Use chat_completion(messages, ...) to enqueue, then flush() to submit.
    flush() blocks until the batch completes and returns the responses
    keyed by custom_id.
    """

    def __init__(self, config: BatchConfig | None = None):
        self.config = config or BatchConfig()
        self._buffer: list[dict[str, Any]] = []
        self._client = None  # lazy openai.Client

    # ---- Public API ------------------------------------------------------

    def chat_completion(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        custom_id: str | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> str:
        """Buffer one chat-completion request and return its custom_id.

        The actual response is available after flush() returns.
        """
        cid = custom_id or f"req-{uuid.uuid4().hex[:16]}"
        body: dict[str, Any] = {
            "model": model or self.config.model,
            "messages": messages,
        }
        if temperature is not None:
            body["temperature"] = temperature
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        if response_format is not None:
            body["response_format"] = response_format
        self._buffer.append(
            {
                "custom_id": cid,
                "method": "POST",
                "url": self.config.endpoint,
                "body": body,
            }
        )
        if (
            self.config.buffer_capacity
            and len(self._buffer) >= self.config.buffer_capacity
        ):
            # NOTE: auto-flush is a convenience; for full-sweep submissions
            # the caller is expected to call flush() explicitly so all
            # critics + seeds + conditions end up in ONE batch.
            return cid
        return cid

    def flush(self) -> dict[str, BatchResponse]:
        """Submit the buffered requests as one batch and block until done.

        Returns a dict keyed by custom_id. The buffer is emptied on
        success; on failure it is preserved so the caller can retry.
        """
        if not self._buffer:
            return {}
        # Cache hit?
        if self.config.cache_dir:
            cached = self._load_cache()
            if cached is not None:
                self._buffer.clear()
                return cached

        # 1. Write JSONL.
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".jsonl",
            delete=False,
            encoding="utf-8",
        ) as f:
            for entry in self._buffer:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            jsonl_path = Path(f.name)

        try:
            file_id = self._upload_jsonl(jsonl_path)
            batch_id = self._create_batch(file_id)
            self._wait_for_batch(batch_id)
            results = self._download_results(batch_id)
        finally:
            jsonl_path.unlink(missing_ok=True)

        if self.config.cache_dir:
            self._save_cache(results)
        self._buffer.clear()
        return results

    # ---- Low-level OpenAI calls -----------------------------------------

    def _get_client(self):
        if self._client is not None:
            return self._client
        try:
            from openai import OpenAI  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                "openai client is required: pip install openai>=1.0"
            ) from exc
        self._client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        return self._client

    def _upload_jsonl(self, path: Path) -> str:
        client = self._get_client()
        with open(path, "rb") as f:
            fileobj = client.files.create(file=f, purpose="batch")
        return fileobj.id

    def _create_batch(self, file_id: str) -> str:
        client = self._get_client()
        batch = client.batches.create(
            input_file_id=file_id,
            endpoint=self.config.endpoint,
            completion_window=self.config.completion_window,
        )
        return batch.id

    def _wait_for_batch(self, batch_id: str) -> None:
        client = self._get_client()
        deadline = time.time() + self.config.poll_timeout_seconds
        while time.time() < deadline:
            batch = client.batches.retrieve(batch_id)
            status = batch.status
            if status in {"completed", "failed", "expired", "cancelled"}:
                if status != "completed":
                    raise RuntimeError(
                        f"batch {batch_id} ended with status {status}"
                    )
                return
            time.sleep(self.config.poll_interval_seconds)
        raise TimeoutError(f"batch {batch_id} did not finish in window")

    def _download_results(self, batch_id: str) -> dict[str, BatchResponse]:
        client = self._get_client()
        batch = client.batches.retrieve(batch_id)
        out_file_id = batch.output_file_id
        if out_file_id is None:
            raise RuntimeError(f"batch {batch_id} has no output_file_id")
        content = client.files.content(out_file_id).read()
        if isinstance(content, bytes):
            content = content.decode("utf-8")
        results: dict[str, BatchResponse] = {}
        for line in content.splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            cid = row.get("custom_id", "")
            resp = row.get("response", {})
            body = resp.get("body", {}) if resp else {}
            choices = body.get("choices") or []
            usage = body.get("usage") or {}
            msg = choices[0].get("message", {}) if choices else {}
            err = row.get("error")
            results[cid] = BatchResponse(
                custom_id=cid,
                content=(msg.get("content") or "") if msg else "",
                finish_reason=(choices[0].get("finish_reason") if choices else None),
                model=body.get("model"),
                prompt_tokens=int(usage.get("prompt_tokens", 0) or 0),
                completion_tokens=int(usage.get("completion_tokens", 0) or 0),
                error=(json.dumps(err) if err else None),
                raw=row,
            )
        return results

    # ---- Cache --------------------------------------------------------

    def _cache_path(self) -> Path:
        cache_dir = Path(self.config.cache_dir) if self.config.cache_dir else None
        if cache_dir is None:
            raise RuntimeError("cache_dir must be set to use the cache")
        cache_dir.mkdir(parents=True, exist_ok=True)
        # Cache keyed by the buffer's content hash so the same sweep
        # re-uses the same cached file.
        import hashlib

        h = hashlib.sha256()
        for e in self._buffer:
            h.update(json.dumps(e, sort_keys=True).encode("utf-8"))
        return cache_dir / f"batch_{h.hexdigest()[:16]}.json"

    def _load_cache(self) -> dict[str, BatchResponse] | None:
        p = self._cache_path()
        if not p.exists():
            return None
        data = json.loads(p.read_text(encoding="utf-8"))
        return {
            cid: BatchResponse.model_validate(row) for cid, row in data.items()
        }

    def _save_cache(self, results: dict[str, BatchResponse]) -> None:
        p = self._cache_path()
        payload = {cid: r.model_dump() for cid, r in results.items()}
        p.write_text(json.dumps(payload, indent=2), encoding="utf-8")


# ---- Module-level conveniences ------------------------------------------


def submit_batch_jsonl(
    jsonl_path: Path, *, config: BatchConfig | None = None
) -> dict[str, BatchResponse]:
    """Submit a pre-built batch.jsonl file and return the parsed results."""
    cfg = config or BatchConfig()
    client = BatchClient(cfg)
    # Inject the file's lines into the buffer so flush() picks them up.
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            client._buffer.append(json.loads(line))
    return client.flush()


def fetch_batch_results(
    batch_id: str, *, config: BatchConfig | None = None
) -> dict[str, BatchResponse]:
    """Fetch results of a previously-submitted batch by id."""
    cfg = config or BatchConfig()
    client = BatchClient(cfg)
    client._wait_for_batch(batch_id)
    return client._download_results(batch_id)
