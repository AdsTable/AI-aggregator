# ai_aggregator.v1.py
"""
AI Aggregator - single-file production-ready prototype with multi-provider support.

Features:
 - Pluggable AI providers (OpenAI + mocks/placeholders for Gemini/DeepSeek/GigaChat/KIMI/Humata/EasyPeasy)
 - Chunking + overlap
 - Lazy summarization (summarize only relevant chunks)
 - Thread-safe JSON cache with atomic writes
 - FAISS incremental index (inner-product on normalized vectors)
 - URL fetcher with pooling and retry/backoff
 - Metrics and basic token tracking
 - Streamlit UI with provider selection
"""

import os
import json
import time
import io
import hashlib
import logging
import threading
import tempfile
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime
from collections import deque
from urllib.parse import urlparse

# Third-party libs
import aiohttp
import streamlit as st
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

import numpy as np
import tiktoken
import faiss
import fitz  # PyMuPDF
import docx
import openpyxl
from bs4 import BeautifulSoup
from tenacity import retry, stop_after_attempt, wait_exponential

# OpenAI async client
from openai import AsyncOpenAI

# nest_asyncio for Streamlit compatibility with asyncio
import nest_asyncio
nest_asyncio.apply()

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("ai_aggregator")

# -------------------- Configuration --------------------
@dataclass
class Config:
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    CACHE_FILE: Path = Path("cache.json")
    FAISS_INDEX_FILE: Path = Path("faiss.index")
    FAISS_MAP_FILE: Path = Path("faiss_map.json")
    EMBED_DIM: int = 1536  # recommended for text-embedding-3-small
    CHUNK_SIZE: int = 900
    CHUNK_OVERLAP: int = 100
    MAX_CONCURRENT_REQUESTS: int = 5
    MAX_FILE_SIZE_MB: int = 100
    MAX_URL_SIZE_MB: int = 10
    CACHE_WRITE_INTERVAL: int = 10
    CONNECTION_TIMEOUT: int = 30
    MAX_RETRIES: int = 3

config = Config()

if not config.OPENAI_API_KEY:
    logger.warning("OPENAI_API_KEY not set — OpenAI provider will fail if selected. Use mocks or set the key in env.")

# Async OpenAI client instance (used by OpenAIProvider)
openai_client = AsyncOpenAI(api_key=config.OPENAI_API_KEY) if config.OPENAI_API_KEY else None

# Pricing (approx USD per 1k tokens)
PRICES = {
    "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    "text-embedding-3-small": {"input": 0.00002, "output": 0.0},
}

# -------------------- Utilities --------------------
def md5_hash_bytes(b: bytes) -> str:
    return hashlib.md5(b).hexdigest()

def get_tiktoken_encoder(model: str = "gpt-3.5-turbo"):
    try:
        return tiktoken.encoding_for_model(model)
    except Exception:
        return tiktoken.get_encoding("cl100k_base")

def truncate_text(text: str, max_tokens: int, model: str = "gpt-3.5-turbo") -> str:
    enc = get_tiktoken_encoder(model)
    toks = enc.encode(text)
    if len(toks) <= max_tokens:
        return text
    return enc.decode(toks[:max_tokens])

def chunk_text(text: str, chunk_size: int = config.CHUNK_SIZE, overlap: int = config.CHUNK_OVERLAP, model: str = "gpt-3.5-turbo") -> List[str]:
    enc = get_tiktoken_encoder(model)
    tokens = enc.encode(text)
    if not tokens:
        return []
    chunks = []
    step = max(1, chunk_size - overlap)
    for i in range(0, len(tokens), step):
        ctoks = tokens[i:i + chunk_size]
        if not ctoks:
            break
        chunks.append(enc.decode(ctoks))
    return chunks

def cosine_sim(a: List[float], b: List[float]) -> float:
    if a is None or b is None:
        return 0.0
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    an = np.linalg.norm(a)
    bn = np.linalg.norm(b)
    if an == 0 or bn == 0:
        return 0.0
    return float(np.dot(a, b) / (an * bn))

# -------------------- Cache Manager --------------------
class CacheManager:
    """Thread-safe JSON cache with atomic writes."""
    def __init__(self, path: Path, write_interval: int = 10):
        self.path = path
        self.write_interval = write_interval
        self.lock = threading.Lock()
        self.cache = self._load()
        self.dirty = False
        self.last_write = time.time()

    def _load(self) -> dict:
        if not self.path.exists():
            return {}
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning("Failed load cache: %s", e)
            return {}

    def get(self, key: str) -> Optional[dict]:
        with self.lock:
            return self.cache.get(key)

    def set(self, key: str, value: dict):
        with self.lock:
            self.cache[key] = value
            self.dirty = True
            if time.time() - self.last_write > self.write_interval:
                self._write()

    def _write(self):
        if not self.dirty:
            return
        tmp = self.path.with_suffix(".tmp")
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=2)
                f.flush()
                os.fsync(f.fileno())
            tmp.replace(self.path)
            self.dirty = False
            self.last_write = time.time()
            logger.info("Cache written to disk")
        except Exception as e:
            logger.error("Cache write failed: %s", e)
            try:
                if tmp.exists():
                    tmp.unlink(missing_ok=True)
            except Exception:
                pass

    def flush(self):
        with self.lock:
            self._write()

# -------------------- FAISS Manager --------------------
class FAISSManager:
    """FAISS manager using inner-product on L2-normalized vectors (IndexFlatIP)."""
    def __init__(self, index_path: Path, map_path: Path, dim: int):
        self.index_path = index_path
        self.map_path = map_path
        self.dim = dim
        self.lock = threading.Lock()
        try:
            if self.index_path.exists() and self.map_path.exists():
                self.index = faiss.read_index(str(self.index_path))
                with open(self.map_path, "r", encoding="utf-8") as f:
                    self.mapping = {int(k): v for k, v in json.load(f).items()}
                logger.info("Loaded FAISS index with %d vectors", self.index.ntotal)
            else:
                self.index = faiss.IndexFlatIP(self.dim)
                self.mapping = {}
        except Exception as e:
            logger.error("Failed loading FAISS: %s", e)
            self.index = faiss.IndexFlatIP(self.dim)
            self.mapping = {}

    def add(self, key: str, embedding: List[float]):
        with self.lock:
            vec = np.array([embedding], dtype=np.float32)
            faiss.normalize_L2(vec)
            self.index.add(vec)
            idx = self.index.ntotal - 1
            self.mapping[idx] = key

    def search(self, embedding: List[float], k: int = 5) -> List[Tuple[str, float]]:
        with self.lock:
            if self.index.ntotal == 0:
                return []
            vec = np.array([embedding], dtype=np.float32)
            faiss.normalize_L2(vec)
            k = min(k, self.index.ntotal)
            D, I = self.index.search(vec, k)
            results = []
            for dist, idx in zip(D[0], I[0]):
                if idx >= 0 and idx in self.mapping:
                    results.append((self.mapping[idx], float(dist)))
            return results

    def save(self):
        with self.lock:
            try:
                faiss.write_index(self.index, str(self.index_path))
                with open(self.map_path, "w", encoding="utf-8") as f:
                    # mapping as {index: cache_key}
                    json.dump({str(k): v for k, v in self.mapping.items()}, f)
                logger.info("FAISS index saved")
            except Exception as e:
                logger.error("Failed saving FAISS: %s", e)

# -------------------- URL Fetcher --------------------
class URLFetcher:
    """Connection-pooled URL fetcher with size checks and retry/backoff."""
    def __init__(self, timeout: int = config.CONNECTION_TIMEOUT, max_size_mb: int = config.MAX_URL_SIZE_MB):
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.max_size = max_size_mb * 1024 * 1024
        self.session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession(timeout=self.timeout, headers={"User-Agent": "ai_aggregator/1.0"})
        return self

    async def __aexit__(self, exc_type, exc, tb):
        if self.session:
            await self.session.close()
            self.session = None

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=5))
    async def fetch(self, url: str) -> Optional[str]:
        try:
            parsed = urlparse(url)
            if parsed.scheme not in ("http", "https"):
                logger.warning("Invalid URL scheme: %s", url)
                return None
            if not parsed.hostname:
                logger.warning("URL without hostname: %s", url)
                return None
            if parsed.hostname in ("127.0.0.1", "localhost", "0.0.0.0"):
                logger.warning("Blocked local host URL: %s", url)
                return None
            assert self.session is not None
            async with self.session.get(url) as resp:
                resp.raise_for_status()
                content_len = resp.headers.get("Content-Length")
                if content_len and int(content_len) > self.max_size:
                    logger.warning("Remote content too large: %s", url)
                    return None
                content = await resp.read()
                if len(content) > self.max_size:
                    content = content[:self.max_size]
                soup = BeautifulSoup(content.decode("utf-8", errors="ignore"), "html.parser")
                for tag in soup(["script", "style", "meta", "link", "noscript"]):
                    tag.decompose()
                text = soup.get_text(separator=" ", strip=True)
                return text[:200000]
        except Exception as e:
            logger.error("URL fetch error %s: %s", url, e)
            return None

# -------------------- OpenAI wrappers --------------------
@retry(stop=stop_after_attempt(config.MAX_RETRIES), wait=wait_exponential(min=1, max=5))
async def openai_get_embedding(text: str, model: str = "text-embedding-3-small") -> Optional[List[float]]:
    if openai_client is None:
        logger.error("OpenAI client not configured")
        return None
    resp = await openai_client.embeddings.create(model=model, input=text)
    usage = getattr(resp, "usage", None)
    tokens = 0
    if usage:
        tokens = getattr(usage, "total_tokens", 0) or getattr(usage, "prompt_tokens", 0) or 0
    if "metrics" in st.session_state:
        st.session_state.metrics.add_api_call(model, tokens, 0)
    return [float(x) for x in resp.data[0].embedding]

@retry(stop=stop_after_attempt(config.MAX_RETRIES), wait=wait_exponential(min=1, max=5))
async def openai_summarize(text: str, model: str, max_input_tokens: int, max_output_tokens: int) -> str:
    if openai_client is None:
        logger.error("OpenAI client not configured")
        return text[:max_output_tokens*2]
    truncated = truncate_text(text, max_input_tokens, model)
    resp = await openai_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a concise summarizer."},
            {"role": "user", "content": truncated},
        ],
        max_tokens=max_output_tokens
    )
    usage = getattr(resp, "usage", None)
    if usage and "metrics" in st.session_state:
        st.session_state.metrics.add_api_call(model, getattr(usage, "prompt_tokens", 0), getattr(usage, "completion_tokens", 0))
    return resp.choices[0].message.content.strip()

# -------------------- Providers abstraction --------------------
class AIProvider:
    """Abstract AI provider interface."""
    async def embed(self, text: str) -> Optional[List[float]]:
        raise NotImplementedError()

    async def summarize(self, text: str, max_input_tokens: int, max_output_tokens: int) -> str:
        raise NotImplementedError()

# Real OpenAI provider using AsyncOpenAI
class OpenAIProvider(AIProvider):
    def __init__(self, client, embedding_model="text-embedding-3-small", summarization_model="gpt-4o-mini"):
        self.client = client
        self.embedding_model = embedding_model
        self.summarization_model = summarization_model

    async def embed(self, text: str) -> Optional[List[float]]:
        return await openai_get_embedding(text, model=self.embedding_model)

    async def summarize(self, text: str, max_input_tokens: int, max_output_tokens: int) -> str:
        return await openai_summarize(text, model=self.summarization_model, max_input_tokens=max_input_tokens, max_output_tokens=max_output_tokens)

# Helper to create deterministic pseudo-embeddings for mocks
def deterministic_vector_from_text(text: str, dim: int = config.EMBED_DIM) -> List[float]:
    # Simple deterministic pseudo-embedding using hashlib and expanding
    h = hashlib.sha256(text.encode("utf-8")).digest()
    rng = np.frombuffer(h * (dim // len(h) + 1), dtype=np.uint8)[:dim].astype(np.float32)
    # Normalize values to [-1,1]
    vec = (rng / 255.0) * 2.0 - 1.0
    # L2 normalize
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec.tolist()
    return (vec / norm).tolist()

# Mock provider examples: these should be replaced with real API clients
class MockProviderBase(AIProvider):
    """Base class for mock providers (returns deterministic embeddings and simple summaries)."""
    def __init__(self, name: str):
        self.name = name

    async def embed(self, text: str) -> Optional[List[float]]:
        # Simulate network latency slightly
        await asyncio.sleep(0.01)
        return deterministic_vector_from_text(self.name + ":" + (text[:1000] if text else ""))

    async def summarize(self, text: str, max_input_tokens: int, max_output_tokens: int) -> str:
        # Lightweight summarization: return first N chars + provider tag
        truncated = truncate_text(text, max_input_tokens, model="gpt-3.5-turbo")
        out = truncated[:max(200, max_output_tokens*2)]
        return f"[{self.name} summary] " + (out[:max_output_tokens*50] if out else "")

# Specific mock providers
class GeminiProvider(MockProviderBase):
    def __init__(self):
        super().__init__("Gemini")

class DeepSeekProvider(MockProviderBase):
    def __init__(self):
        super().__init__("DeepSeek")

class GigaChatProvider(MockProviderBase):
    def __init__(self):
        super().__init__("GigaChat")

class KIMIProvider(MockProviderBase):
    def __init__(self):
        super().__init__("KIMI")

class HumataProvider(MockProviderBase):
    def __init__(self):
        super().__init__("Humata")

class EasyPeasyProvider(MockProviderBase):
    def __init__(self):
        super().__init__("EasyPeasy")

# Provider factory
def get_provider_by_name(name: str) -> AIProvider:
    name = name.lower()
    if name == "openai":
        return OpenAIProvider(openai_client)
    if name == "gemini":
        return GeminiProvider()
    if name == "deepseek":
        return DeepSeekProvider()
    if name == "gigachat":
        return GigaChatProvider()
    if name == "kimi":
        return KIMIProvider()
    if name == "humata":
        return HumataProvider()
    if name == "easypeasy":
        return EasyPeasyProvider()
    # default fallback
    return MockProviderBase("fallback")

# -------------------- File parsers (safe) --------------------
def read_csv_safe(b: bytes, max_rows: int = 10000) -> str:
    try:
        df = pd.read_csv(io.BytesIO(b), nrows=max_rows)
        return df.to_string()
    except Exception:
        return b.decode("utf-8", errors="ignore")[:50000]

def read_xlsx_safe(b: bytes, max_rows: int = 10000) -> str:
    try:
        wb = openpyxl.load_workbook(io.BytesIO(b), read_only=True, data_only=True)
        out = []
        for sheet in wb.worksheets:
            out.append(f"--- Sheet: {sheet.title} ---")
            i = 0
            for row in sheet.iter_rows(values_only=True):
                if i >= max_rows:
                    out.append(f"... truncated at {max_rows} rows")
                    break
                out.append(" ".join([str(c) for c in row if c is not None]))
                i += 1
        return "\n".join(out)
    except Exception:
        return b.decode("utf-8", errors="ignore")[:50000]

def read_pdf_safe(b: bytes, max_pages: int = 100) -> str:
    try:
        doc = fitz.open(stream=b, filetype="pdf")
        texts = []
        for i, p in enumerate(doc):
            if i >= max_pages:
                texts.append(f"... truncated at {max_pages} pages")
                break
            texts.append(p.get_text() or "")
        return "\n".join(texts)
    except Exception:
        return ""

def read_docx_safe(b: bytes) -> str:
    try:
        doc = docx.Document(io.BytesIO(b))
        return "\n".join([p.text for p in doc.paragraphs if p.text])
    except Exception:
        return b.decode("utf-8", errors="ignore")[:50000]

def parse_file(path: Path) -> Optional[str]:
    try:
        size_mb = path.stat().st_size / (1024 * 1024)
    except Exception:
        size_mb = 0
    if size_mb > config.MAX_FILE_SIZE_MB:
        logger.warning("File too large: %s (%.2f MB)", path.name, size_mb)
        return None
    b = path.read_bytes()
    ext = path.suffix.lower()
    if ext == ".csv":
        return read_csv_safe(b)
    if ext == ".json":
        try:
            return json.dumps(json.loads(b), ensure_ascii=False, indent=2)
        except Exception:
            return b.decode("utf-8", errors="ignore")[:50000]
    if ext == ".pdf":
        return read_pdf_safe(b)
    if ext in (".docx", ".doc"):
        return read_docx_safe(b)
    if ext in (".xlsx", ".xls", ".xlsm"):
        return read_xlsx_safe(b)
    return b.decode("utf-8", errors="ignore")[:100000]

# -------------------- Metrics --------------------
class Metrics:
    def __init__(self):
        self.input_tokens = 0
        self.output_tokens = 0
        self.cost = 0.0
        self.api_calls = 0
        self.cache_hits = 0
        self.errors = 0
        self.start = time.time()

    def add_api_call(self, model: str, input_toks: int = 0, output_toks: int = 0, is_cached: bool = False):
        if is_cached:
            self.cache_hits += 1
            return
        self.api_calls += 1
        self.input_tokens += int(input_toks)
        self.output_tokens += int(output_toks)
        price = PRICES.get(model, {"input": 0.0, "output": 0.0})
        self.cost += (input_toks / 1000.0) * price["input"] + (output_toks / 1000.0) * price["output"]

    def add_error(self):
        self.errors += 1

    def get(self) -> Dict[str, Any]:
        elapsed = time.time() - self.start
        total_tokens = self.input_tokens + self.output_tokens
        hit_rate = self.cache_hits / max(1, self.cache_hits + self.api_calls)
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": total_tokens,
            "cost_usd": round(self.cost, 6),
            "api_calls": self.api_calls,
            "cache_hits": self.cache_hits,
            "cache_hit_rate": round(hit_rate, 3),
            "errors": self.errors,
            "elapsed_seconds": round(elapsed, 2),
            "tokens_per_second": round(total_tokens / max(1, elapsed), 2) if elapsed > 0 else 0,
        }

# -------------------- Processing pipeline --------------------
async def process_chunk_pipeline(
    chunk: str,
    chunk_hash: str,
    resource_name: str,
    query_embedding: List[float],
    cache_mgr: CacheManager,
    faiss_mgr: FAISSManager,
    provider: AIProvider,
    semaphore: asyncio.Semaphore,
    model_name_for_summary: str,
    max_input_tokens: int,
    max_output_tokens: int,
    relevancy_threshold: float,
) -> Optional[Dict[str, Any]]:
    """
    Process a single chunk: check cache, embed, (maybe) summarize, update cache/FAISS.
    Return dict if relevant, else None.
    """
    cached = cache_mgr.get(chunk_hash)
    if cached and cached.get("embedding"):
        sim = cosine_sim(cached["embedding"], query_embedding)
        if sim >= relevancy_threshold and cached.get("summary"):
            if "metrics" in st.session_state:
                st.session_state.metrics.add_api_call(model_name_for_summary, 0, 0, is_cached=True)
            return {"resource_name": cached.get("resource_name", resource_name), "summary": cached.get("summary"), "similarity": sim}

    # Embed (under semaphore)
    async with semaphore:
        emb = await provider.embed(chunk)
    if not emb:
        return None
    sim = cosine_sim(emb, query_embedding)
    entry = {"embedding": emb, "resource_name": resource_name, "timestamp": datetime.now().isoformat()}

    if sim >= relevancy_threshold:
        # Summarize (under semaphore)
        async with semaphore:
            summary = await provider.summarize(chunk, max_input_tokens, max_output_tokens)
        entry["summary"] = summary
        faiss_mgr.add(chunk_hash, emb)
        result = {"resource_name": resource_name, "summary": summary, "similarity": sim}
    else:
        entry["summary"] = ""
        result = None

    cache_mgr.set(chunk_hash, entry)
    return result

async def process_single_resource(
    resource: Union[str, Path],
    query_embedding: List[float],
    provider: AIProvider,
    model_name_for_summary: str,
    max_input_tokens: int,
    max_output_tokens: int,
    cache_mgr: CacheManager,
    faiss_mgr: FAISSManager,
    semaphore: asyncio.Semaphore,
    url_fetcher: URLFetcher,
    relevancy_threshold: float,
) -> List[Dict[str, Any]]:
    """Fetch/parse a single resource and process its chunks."""
    resource_name = str(resource)
    try:
        if isinstance(resource, Path):
            content = parse_file(resource)
        else:
            content = await url_fetcher.fetch(resource)
        if not content:
            return []
        chunks = chunk_text(content, config.CHUNK_SIZE, config.CHUNK_OVERLAP, model_name_for_summary)
        tasks = []
        for chunk in chunks:
            chash = md5_hash_bytes(chunk.encode("utf-8"))
            tasks.append(process_chunk_pipeline(chunk, chash, resource_name, query_embedding, cache_mgr, faiss_mgr, provider, semaphore, model_name_for_summary, max_input_tokens, max_output_tokens, relevancy_threshold))
        chunk_results = await asyncio.gather(*tasks, return_exceptions=True)
        results = []
        for cr in chunk_results:
            if isinstance(cr, Exception):
                logger.error("Chunk processing error: %s", cr)
                if "metrics" in st.session_state:
                    st.session_state.metrics.add_error()
                continue
            if cr:
                results.append(cr)
        return results
    except Exception as e:
        logger.error("Failed processing resource %s: %s", resource, e)
        if "metrics" in st.session_state:
            st.session_state.metrics.add_error()
        return []

async def execute_query(
    query: str,
    files: List[Union[str, Path]],
    urls: List[str],
    provider: AIProvider,
    model_name_for_summary: str,
    max_input_tokens: int,
    max_output_tokens: int,
    relevancy_threshold: float,
) -> Dict[str, Any]:
    """Orchestrator: embed query, consult FAISS, process new resources, return results + metrics."""
    if "metrics" not in st.session_state:
        st.session_state.metrics = Metrics()

    # get query embedding (single call)
    query_emb = await provider.embed(query)
    if not query_emb:
        return {"query": query, "results": [], "metrics": st.session_state.metrics.get(), "error": "Failed to embed query"}

    cache_mgr = CacheManager(config.CACHE_FILE, config.CACHE_WRITE_INTERVAL)
    faiss_mgr = FAISSManager(config.FAISS_INDEX_FILE, config.FAISS_MAP_FILE, config.EMBED_DIM)
    semaphore = asyncio.Semaphore(config.MAX_CONCURRENT_REQUESTS)

    # quick hits from FAISS
    faiss_hits = faiss_mgr.search(query_emb, k=20)
    quick_hits = []
    processed_resources = set()

    for key, sim in faiss_hits:
        if sim >= relevancy_threshold:
            cached = cache_mgr.get(key)
            if cached and cached.get("summary"):
                resource_name = cached.get("resource_name")
                if resource_name:
                    processed_resources.add(resource_name)
                quick_hits.append({"resource_name": resource_name, "summary": cached.get("summary"), "similarity": sim})
                if "metrics" in st.session_state:
                    st.session_state.metrics.add_api_call(model_name_for_summary, 0, 0, is_cached=True)

    # Prepare list of resources not already covered
    all_resources = [Path(p) if isinstance(p, str) and os.path.exists(p) else p for p in files] + list(urls)
    resources_to_process = [r for r in all_resources if str(r) not in processed_resources]

    tasks = []
    async with URLFetcher(config.CONNECTION_TIMEOUT, config.MAX_URL_SIZE_MB) as fetcher:
        for resource in resources_to_process:
            tasks.append(process_single_resource(resource, query_emb, provider, model_name_for_summary, max_input_tokens, max_output_tokens, cache_mgr, faiss_mgr, semaphore, fetcher, relevancy_threshold))
        new_results_nested = await asyncio.gather(*tasks, return_exceptions=True)

    results = quick_hits
    for item in new_results_nested:
        if isinstance(item, Exception):
            logger.error("Batch processing error: %s", item)
            continue
        results.extend(item)

    cache_mgr.flush()
    faiss_mgr.save()

    # deduplicate and sort
    unique = {}
    for r in results:
        key = (r["resource_name"], r.get("summary", "")[:200])
        if key not in unique or unique[key]["similarity"] < r["similarity"]:
            unique[key] = r
    out = sorted(unique.values(), key=lambda x: x["similarity"], reverse=True)[:50]

    return {"query": query, "results": out, "metrics": st.session_state.metrics.get()}

# -------------------- Streamlit UI helpers --------------------
def init_state():
    if "metrics" not in st.session_state:
        st.session_state.metrics = Metrics()
    if "query_history" not in st.session_state:
        st.session_state.query_history = deque(maxlen=50)
    if "results_cache" not in st.session_state:
        st.session_state.results_cache = {}

def display_results(results: List[Dict[str, Any]]):
    if not results:
        st.info("No relevant results found.")
        return
    for r in results:
        st.markdown(f"**{r['resource_name']}** — similarity: {r['similarity']:.3f}")
        st.write(r["summary"][:2000])
        st.divider()

def run_async(coro):
    """Run coroutine safely inside Streamlit environment (nest_asyncio applied)."""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    if loop.is_running():
        # run_until_complete works with nest_asyncio applied
        return loop.run_until_complete(coro)
    else:
        return loop.run_until_complete(coro)

# -------------------- Main UI --------------------
def main():
    st.set_page_config(page_title="AI Aggregator (Multi-provider)", layout="wide")
    init_state()
    st.title("🔎 AI Aggregator — Multi-provider (prototype)")

    # Sidebar: provider + settings
    with st.sidebar:
        st.header("Provider & settings")

        provider_name = st.selectbox("AI provider", ["OpenAI", "Gemini", "DeepSeek", "GigaChat", "KIMI", "Humata", "EasyPeasy"])
        # for OpenAI we allow selecting summarization model; for mocks it's ignored
        if provider_name == "OpenAI":
            summarization_model = st.selectbox("OpenAI Summarization model", ["gpt-4o-mini", "gpt-3.5-turbo"])
            embedding_model = st.text_input("OpenAI Embedding model", value="text-embedding-3-small")
        else:
            summarization_model = st.text_input("Summarization model", value="(provider default)")
            embedding_model = st.text_input("Embedding model", value="(provider default)")

        relevancy_threshold = st.slider("Relevancy threshold", 0.0, 1.0, 0.6, 0.01)
        max_input_tokens = st.number_input("Max input tokens (per chunk)", value=1200, min_value=100)
        max_output_tokens = st.number_input("Max output tokens (summary)", value=256, min_value=50)

        if st.button("Clear cache & index"):
            try:
                config.CACHE_FILE.unlink(missing_ok=True)
                config.FAISS_INDEX_FILE.unlink(missing_ok=True)
                config.FAISS_MAP_FILE.unlink(missing_ok=True)
                st.success("Cleared cache and FAISS index (files removed).")
            except Exception as e:
                st.error(f"Failed to clear: {e}")

    # Main: query, files, urls
    query = st.text_input("Query", placeholder="Find cheapest electricity providers in Oslo")
    uploaded_files = st.file_uploader("Upload files", accept_multiple_files=True, type=["pdf","docx","txt","csv","json","xlsx"])
    urls_text = st.text_area("URLs (one per line)", placeholder="https://example.com/info\nhttps://another.example")
    urls = [u.strip() for u in urls_text.splitlines() if u.strip()]

    # Provider instance
    provider = get_provider_by_name(provider_name)

    if st.button("Run"):
        valid = True
        if not query:
            st.error("Please enter a query.")
            valid = False
        if not uploaded_files and not urls:
            st.warning("No files or URLs supplied — the query will only use provider semantics (may return limited results).")
        if valid:
            st.session_state.query_history.append(query)
            with tempfile.TemporaryDirectory() as tmpdir:
                local_paths: List[str] = []
                for f in uploaded_files:
                    p = Path(tmpdir) / f.name
                    p.write_bytes(f.read())
                    local_paths.append(str(p))

                with st.spinner("Processing..."):
                    result = run_async(execute_query(query, local_paths, urls, provider, summarization_model, max_input_tokens, max_output_tokens, relevancy_threshold))

                if "error" in result:
                    st.error(result["error"])
                else:
                    st.subheader("Results")
                    display_results(result["results"])
                    st.subheader("Metrics")
                    st.json(result["metrics"])

    # Footer: recent queries
    if st.session_state.query_history:
        st.sidebar.markdown("### Recent queries")
        for q in list(reversed(st.session_state.query_history))[:10]:
            st.sidebar.write(q)

if __name__ == "__main__":
    main()
