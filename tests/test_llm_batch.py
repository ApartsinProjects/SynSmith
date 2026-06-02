"""Unit tests for the OpenAI Batch API harness.

These tests do NOT hit the OpenAI API. They verify:

1. chat_completion() buffers requests with valid OpenAI Batch schema.
2. Custom_ids are unique per call.
3. The batch JSONL is well-formed (one valid JSON object per line, each
   with custom_id / method / url / body fields per OpenAI's spec).
4. Cache load/save round-trips correctly.
5. submit_batch_jsonl reads a pre-built JSONL into the buffer.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from attrforge.llm_batch import (
    BatchClient,
    BatchConfig,
    BatchResponse,
    submit_batch_jsonl,
)


def test_buffer_one_request():
    client = BatchClient(BatchConfig(model="gpt-4o-mini"))
    cid = client.chat_completion(
        [{"role": "user", "content": "hello"}],
        temperature=0.7,
        max_tokens=128,
    )
    assert cid.startswith("req-")
    assert len(client._buffer) == 1
    entry = client._buffer[0]
    assert entry["custom_id"] == cid
    assert entry["method"] == "POST"
    assert entry["url"] == "/v1/chat/completions"
    body = entry["body"]
    assert body["model"] == "gpt-4o-mini"
    assert body["temperature"] == 0.7
    assert body["max_tokens"] == 128
    assert body["messages"][0]["content"] == "hello"


def test_unique_custom_ids():
    """Each enqueued call should get a unique custom_id by default."""
    client = BatchClient(BatchConfig())
    cids = [
        client.chat_completion([{"role": "user", "content": f"q{i}"}])
        for i in range(20)
    ]
    assert len(set(cids)) == 20


def test_custom_id_can_be_overridden():
    """A caller-supplied custom_id is preserved verbatim."""
    client = BatchClient(BatchConfig())
    cid = client.chat_completion(
        [{"role": "user", "content": "q"}],
        custom_id="my-sample-001",
    )
    assert cid == "my-sample-001"
    assert client._buffer[0]["custom_id"] == "my-sample-001"


def test_buffer_json_serializable():
    """Each buffered entry must serialize to JSONL the Batch API accepts."""
    client = BatchClient(BatchConfig())
    for i in range(5):
        client.chat_completion(
            [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": f"question {i}"},
            ],
            temperature=0.5,
        )
    # Every entry should round-trip JSON cleanly and parse back identical.
    for entry in client._buffer:
        text = json.dumps(entry, ensure_ascii=False)
        parsed = json.loads(text)
        assert parsed == entry


def test_cache_round_trip(tmp_path):
    """A saved cache should reload identically on next flush() call."""
    cfg = BatchConfig(cache_dir=tmp_path)
    client = BatchClient(cfg)
    client._buffer = [
        {
            "custom_id": "x1",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "q"}]},
        }
    ]
    fake = {"x1": BatchResponse(custom_id="x1", content="hello world", prompt_tokens=5, completion_tokens=2)}
    client._save_cache(fake)

    # Fresh client with same buffer + cache_dir should hit cache.
    client2 = BatchClient(cfg)
    client2._buffer = list(client._buffer)
    loaded = client2._load_cache()
    assert loaded is not None
    assert loaded["x1"].content == "hello world"
    assert loaded["x1"].prompt_tokens == 5


def test_submit_batch_jsonl_loads_file(tmp_path, monkeypatch):
    """submit_batch_jsonl reads the file into a fresh BatchClient buffer.

    We monkey-patch flush() to skip the actual API call so the test stays
    offline.
    """
    p = tmp_path / "in.jsonl"
    p.write_text(
        json.dumps(
            {
                "custom_id": "r1",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "q"}]},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    seen_buffer: list[list[dict]] = []

    def fake_flush(self):
        seen_buffer.append(list(self._buffer))
        return {}

    monkeypatch.setattr(BatchClient, "flush", fake_flush, raising=True)
    submit_batch_jsonl(p)
    assert seen_buffer
    assert seen_buffer[0][0]["custom_id"] == "r1"
