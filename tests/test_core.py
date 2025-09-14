# tests/test_core.py
import sys
import os
import pytest
import numpy as np
import asyncio
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ai_aggregator import (
    chunk_text, cosine_sim, CacheManager, FAISSManager,
    process_chunk_pipeline, process_single_resource,
    execute_query, md5_hash_bytes, get_provider_by_name
)

# -------------------- Tests --------------------

@pytest.mark.asyncio
async def test_chunk_and_cosine():
    text = "word " * 300
    chunks = chunk_text(text, chunk_size=50, overlap=10)
    assert len(chunks) > 1
    v1 = np.random.rand(16).astype(np.float32)
    v2 = v1.copy()
    assert cosine_sim(v1, v2) > 0.99

def test_cache_manager(tmp_path):
    cache_file = tmp_path / "cache.json"
    cm = CacheManager(cache_file, write_interval=0)
    cm.set("k1", {"val": 123})
    cm.flush()
    cm2 = CacheManager(cache_file)
    assert cm2.get("k1")["val"] == 123

def test_faiss_manager(tmp_path):
    index_file = tmp_path / "index.faiss"
    map_file = tmp_path / "map.json"
    fm = FAISSManager(index_file, map_file, 16)
    emb = np.random.rand(16).astype(np.float32).tolist()
    fm.add("key1", emb)
    results = fm.search(emb, k=1)
    assert results
    fm.save()
    fm2 = FAISSManager(index_file, map_file, 16)
    results2 = fm2.search(emb, k=1)
    assert results2

class MockProvider:
    async def get_embedding(self, text, model=None):
        return [0.0] * 16  # dummy embedding

    async def embed(self, text, model=None):
        return await self.get_embedding(text, model)

    async def summarize(self, text, model=None, max_tokens=None):
        return "summary"

    async def chat(self, messages, model=None, max_tokens=None, temperature=None):
        return "chat response"

@pytest.mark.asyncio
async def test_execute_query_mock():
    # Minimal startup without files and URLs
    provider = MockProvider()
    result = await execute_query(
        "What is AI?",
        [],
        [],
        provider,
        "gpt-3.5-turbo",
        500,
        50,
        0.3,
    )
    assert "query" in result
    assert "results" in result
    assert "metrics" in result

def test_md5_hash_bytes():
    h1 = md5_hash_bytes(b"hello")
    h2 = md5_hash_bytes(b"hello")
    assert h1 == h2
