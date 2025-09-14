# app.py
"""
AI Aggregator - Production-ready Streamlit single-file app
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

import aiohttp
import streamlit as st
import pandas as pd
import numpy as np
import tiktoken
import faiss
import fitz  # PyMuPDF
import docx
import openpyxl
from bs4 import BeautifulSoup
from tenacity import retry, stop_after_attempt, wait_exponential

from openai import AsyncOpenAI
import nest_asyncio
nest_asyncio.apply()

# Logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("ai-aggregator")

# ----------------------- Configuration dataclass -----------------------
@dataclass
class Config:
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    CACHE_FILE: Path = Path("cache.json")
    FAISS_INDEX_FILE: Path = Path("faiss.index")
    FAISS_MAP_FILE: Path = Path("faiss_map.json")
    EMBED_DIM: int = 1536  # text-embedding-3-small dimension
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
    raise RuntimeError("OPENAI_API_KEY not set in environment")

# Async OpenAI client
client = AsyncOpenAI(api_key=config.OPENAI_API_KEY)

# Pricing (USD per 1k tokens) - approximate
PRICES = {
    "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    "text-embedding-3-small": {"input": 0.00002, "output": 0.0},
}

# ----------------------- Utilities -----------------------
def md5_hash_bytes(b: bytes) -> str:
    """Return MD5 hash of bytes."""
    return hashlib.md5(b).hexdigest()

def get_tiktoken_encoder(model: str = "gpt-3.5-turbo"):
    """Return tiktoken encoder for model or fallback."""
    try:
        return tiktoken.encoding_for_model(model)
    except Exception:
        return tiktoken.get_encoding("cl100k_base")

def truncate_text(text: str, max_tokens: int, model: str = "gpt-3.5-turbo") -> str:
    """Truncate text to max_tokens using tiktoken encoder."""
    enc = get_tiktoken_encoder(model)
    toks = enc.encode(text)
    if len(toks) <= max_tokens:
        return text
    return enc.decode(toks[:max_tokens])

def chunk_text(text: str, chunk_size: int = config.CHUNK_SIZE,
               overlap: int = config.CHUNK_OVERLAP, model: str = "gpt-3.5-turbo") -> List[str]:
    """Split text into token-based chunks with overlap."""
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
    """Compute cosine similarity between two vectors."""
    if a is None or b is None:
        return 0.0
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

# ----------------------- Cache Manager -----------------------
class CacheManager:
    """Thread-safe JSON cache manager with atomic writes."""
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
            logger.warning(f"Failed load cache: {e}")
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
        """Atomic write to disk."""
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
            logger.error(f"Cache write failed: {e}")
            if tmp.exists():
                try:
                    tmp.unlink(missing_ok=True)
                except Exception:
                    pass

    def flush(self):
        with self.lock:
            self._write()

# ----------------------- FAISS Manager -----------------------
class FAISSManager:
    """Thread-safe FAISS manager using inner-product on L2-normalized vectors."""
    def __init__(self, index_path: Path, map_path: Path, dim: int):
        self.index_path = index_path
        self.map_path = map_path
        self.dim = dim
        self.lock = threading.Lock()
        try:
            if self.index_path.exists() and self.map_path.exists():
                self.index = faiss.read_index(str(self.index_path))
                with open(self.map_path, "r", encoding="utf-8") as f:
                    # mapping stored as {index_str: cache_key}
                    self.mapping = {int(k): v for k, v in json.load(f).items()}
                logger.info(f"Loaded FAISS index with {self.index.ntotal} vectors")
            else:
                self.index = faiss.IndexFlatIP(self.dim)
                self.mapping = {}
        except Exception as e:
            logger.error(f"Failed loading FAISS: {e}, creating new index")
            self.index = faiss.IndexFlatIP(self.dim)
            self.mapping = {}

    def add(self, key: str, embedding: List[float]):
        """Normalize embedding and add to index."""
        with self.lock:
            vec = np.array([embedding], dtype=np.float32)
            faiss.normalize_L2(vec)
            self.index.add(vec)
            idx = self.index.ntotal - 1
            self.mapping[idx] = key

    def search(self, embedding: List[float], k: int = 5) -> List[Tuple[str, float]]:
        """Search and return list of (cache_key, similarity)"""
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
                # FIX: correctly write mapping as {index: cache_key}
                with open(self.map_path, "w", encoding="utf-8") as f:
                    json.dump({str(k): v for k, v in self.mapping.items()}, f)
                logger.info("FAISS index and mapping saved")
            except Exception as e:
                logger.error(f"Failed saving FAISS: {e}")

# ----------------------- URL Fetcher -----------------------
class URLFetcher:
    """Connection-pooled URL fetcher with size checks and retry/backoff."""
    def __init__(self, timeout: int = config.CONNECTION_TIMEOUT, max_size_mb: int = config.MAX_URL_SIZE_MB):
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.max_size = max_size_mb * 1024 * 1024
        self.session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession(timeout=self.timeout, headers={"User-Agent": "AI-Aggregator/1.0"})
        return self

    async def __aexit__(self, exc_type, exc, tb):
        if self.session:
            await self.session.close()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=5))
    async def fetch(self, url: str) -> Optional[str]:
        try:
            parsed = urlparse(url)
            if parsed.scheme not in ("http", "https"):
                logger.warning(f"Invalid URL scheme for {url}")
                return None
            if not parsed.hostname or parsed.hostname in ("127.0.0.1", "localhost", "0.0.0.0"):
                logger.warning(f"Blocked local URL {url}")
                return None

            assert self.session is not None
            async with self.session.get(url) as resp:
                resp.raise_for_status()
                content_len = resp.headers.get("Content-Length")
                if content_len and int(content_len) > self.max_size:
                    logger.warning(f"Remote content too large from {url}")
                    return None
                content = await resp.read()
                if len(content) > self.max_size:
                    content = content[:self.max_size]
                soup = BeautifulSoup(content.decode("utf-8", errors="ignore"), "html.parser")
                for tag in soup(["script", "style", "meta", "link", "noscript"]):
                    tag.decompose()
                text = soup.get_text(separator=" ", strip=True)
                return text[:100000]  # cap
        except Exception as e:
            logger.error(f"URL fetch error {url}: {e}")
            return None

# ----------------------- OpenAI wrappers -----------------------
@retry(stop=stop_after_attempt(config.MAX_RETRIES), wait=wait_exponential(min=1, max=5))
async def openai_get_embedding(text: str, model: str = "text-embedding-3-small") -> Optional[List[float]]:
    """Get embedding from OpenAI with retry/backoff."""
    resp = await client.embeddings.create(model=model, input=text)
    usage = getattr(resp, "usage", None)
    tokens = 0
    if usage:
        tokens = getattr(usage, "total_tokens", 0) or getattr(usage, "prompt_tokens", 0) or 0
    if "metrics" in st.session_state:
        st.session_state.metrics.add_api_call(model, tokens, 0)
    return [float(x) for x in resp.data[0].embedding]

@retry(stop=stop_after_attempt(config.MAX_RETRIES), wait=wait_exponential(min=1, max=5))
async def openai_summarize(text: str, model: str, max_input_tokens: int, max_output_tokens: int) -> str:
    """Summarize text via OpenAI ChatCompletion with retry."""
    truncated = truncate_text(text, max_input_tokens, model)
    resp = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a concise summarizer."},
            {"role": "user", "content": truncated}
        ],
        max_tokens=max_output_tokens
    )
    usage = getattr(resp, "usage", None)
    if usage and "metrics" in st.session_state:
        st.session_state.metrics.add_api_call(model, getattr(usage, "prompt_tokens", 0), getattr(usage, "completion_tokens", 0))
    return resp.choices[0].message.content.strip()

# ----------------------- File parsers (safe) -----------------------
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
    """Parse file content with size limits and safe handlers."""
    try:
        size_mb = path.stat().st_size / (1024*1024)
    except Exception:
        size_mb = 0
    if size_mb > config.MAX_FILE_SIZE_MB:
        logger.warning(f"File {path.name} too large: {size_mb:.2f}MB")
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

# ----------------------- Metrics -----------------------
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
        self.cost += (input_toks/1000.0) * price["input"] + (output_toks/1000.0) * price["output"]

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
            "cost_usd": self.cost,
            "api_calls": self.api_calls,
            "cache_hits": self.cache_hits,
            "cache_hit_rate": hit_rate,
            "errors": self.errors,
            "elapsed_seconds": elapsed,
            "tokens_per_second": total_tokens / max(1, elapsed) if elapsed > 0 else 0
        }

# ----------------------- Processing pipeline -----------------------
async def process_chunk_pipeline(chunk: str, chunk_hash: str, resource_name: str,
                                 query_embedding: List[float],
                                 cache_mgr: CacheManager, faiss_mgr: FAISSManager,
                                 semaphore: asyncio.Semaphore, model: str,
                                 max_input_tokens: int, max_output_tokens: int,
                                 relevancy_threshold: float) -> Optional[Dict[str,Any]]:
    """
    Process a single chunk: check cache, embed, (maybe) summarize, update cache/FAISS.
    Returns a result dict if chunk is relevant, otherwise None.
    """
    cached = cache_mgr.get(chunk_hash)
    if cached and cached.get("embedding"):
        sim = cosine_sim(cached["embedding"], query_embedding)
        if sim >= relevancy_threshold and cached.get("summary"):
            if "metrics" in st.session_state:
                st.session_state.metrics.add_api_call(model, 0, 0, is_cached=True)
            return {"resource_name": cached.get("resource_name", resource_name), "summary": cached.get("summary"), "similarity": sim}

    # Acquire semaphore for embedding
    async with semaphore:
        emb = await openai_get_embedding(chunk)
    if not emb:
        return None
    sim = cosine_sim(emb, query_embedding)
    entry = {"embedding": emb, "resource_name": resource_name, "timestamp": datetime.now().isoformat()}

    if sim >= relevancy_threshold:
        # If relevant, get summary (protected by semaphore again to limit concurrency)
        async with semaphore:
            summary = await openai_summarize(chunk, model, max_input_tokens, max_output_tokens)
        entry["summary"] = summary
        # Update FAISS and cache
        faiss_mgr.add(chunk_hash, emb)
        result = {"resource_name": resource_name, "summary": summary, "similarity": sim}
    else:
        entry["summary"] = ""
        result = None

    cache_mgr.set(chunk_hash, entry)
    return result

async def process_single_resource(resource: Union[str, Path], query_embedding: List[float],
                                  model: str, max_input_tokens: int, max_output_tokens: int,
                                  cache_mgr: CacheManager, faiss_mgr: FAISSManager,
                                  semaphore: asyncio.Semaphore, url_fetcher: URLFetcher,
                                  relevancy_threshold: float) -> List[Dict[str,Any]]:
    """
    Fetch/parse a single resource and process its chunks.
    Returns list of result dicts for relevant chunks.
    """
    resource_name = str(resource)
    try:
        if isinstance(resource, Path):
            content = parse_file(resource)
        else:
            content = await url_fetcher.fetch(resource)
        if not content:
            return []

        chunks = chunk_text(content, config.CHUNK_SIZE, config.CHUNK_OVERLAP, model)
        tasks = []
        for chunk in chunks:
            chash = md5_hash_bytes(chunk.encode("utf-8"))
            tasks.append(process_chunk_pipeline(chunk, chash, resource_name, query_embedding,
                                                cache_mgr, faiss_mgr, semaphore, model,
                                                max_input_tokens, max_output_tokens, relevancy_threshold))
        chunk_results = await asyncio.gather(*tasks, return_exceptions=True)
        results = []
        for cr in chunk_results:
            if isinstance(cr, Exception):
                logger.error(f"Chunk processing error: {cr}")
                if "metrics" in st.session_state:
                    st.session_state.metrics.add_error()
                continue
            if cr:
                results.append(cr)
        return results
    except Exception as e:
        logger.error(f"Failed processing resource {resource}: {e}")
        if "metrics" in st.session_state:
            st.session_state.metrics.add_error()
        return []

async def execute_query(query: str, files: List[Union[str,Path]], urls: List[str], model: str,
                        max_input_tokens: int, max_output_tokens: int, relevancy_threshold: float) -> Dict[str,Any]:
    """
    Top-level orchestrator: get query embed, consult FAISS, process batches, return results + metrics.
    """
    if "metrics" not in st.session_state:
        st.session_state.metrics = Metrics()
    query_emb = await openai_get_embedding(query)
    if not query_emb:
        return {"query": query, "results": [], "metrics": st.session_state.metrics.get(), "error": "Failed to embed query"}

    cache_mgr = CacheManager(config.CACHE_FILE, config.CACHE_WRITE_INTERVAL)
    faiss_mgr = FAISSManager(config.FAISS_INDEX_FILE, config.FAISS_MAP_FILE, config.EMBED_DIM)
    semaphore = asyncio.Semaphore(config.MAX_CONCURRENT_REQUESTS)

    # Quick lookup from FAISS for cached relevant chunks
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
                    st.session_state.metrics.add_api_call(model, 0, 0, is_cached=True)

    # Prepare resource list (files & urls) excluding those already covered by FAISS hits
    all_resources = [Path(p) if isinstance(p, str) and os.path.exists(p) else p for p in files] + list(urls)
    resources_to_process = [r for r in all_resources if str(r) not in processed_resources]

    tasks = []
    async with URLFetcher(config.CONNECTION_TIMEOUT, config.MAX_URL_SIZE_MB) as fetcher:
        for resource in resources_to_process:
            tasks.append(process_single_resource(resource, query_emb, model, max_input_tokens, max_output_tokens, cache_mgr, faiss_mgr, semaphore, fetcher, relevancy_threshold))
        new_results_nested = await asyncio.gather(*tasks, return_exceptions=True)

    results = quick_hits
    for result_list in new_results_nested:
        if isinstance(result_list, Exception):
            logger.error(f"Batch processing error: {result_list}")
            continue
        results.extend(result_list)

    # Persist caches and index
    cache_mgr.flush()
    faiss_mgr.save()

    # Deduplicate by (resource, summary-prefix) and pick best similarity
    unique = {}
    for r in results:
        key = (r["resource_name"], r.get("summary", "")[:200])
        if key not in unique or unique[key]["similarity"] < r["similarity"]:
            unique[key] = r
    out = sorted(unique.values(), key=lambda x: x["similarity"], reverse=True)[:50]

    return {"query": query, "results": out, "metrics": st.session_state.metrics.get()}

# ----------------------- Streamlit UI -----------------------
def init_state():
    if "metrics" not in st.session_state:
        st.session_state.metrics = Metrics()
    if "query_history" not in st.session_state:
        st.session_state.query_history = deque(maxlen=20)
    if "results_cache" not in st.session_state:
        st.session_state.results_cache = {}

def display_results(results: List[Dict[str,Any]]):
    if not results:
        st.info("No relevant results found.")
        return
    for r in results:
        st.markdown(f"**{r['resource_name']}** — similarity: {r['similarity']:.3f}")
        st.write(r["summary"][:1000])
        st.divider()

def run_async(coro):
    """Run coroutine safely inside Streamlit's environment."""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    if loop.is_running():
        # nest_asyncio applied at top; run coroutine on running loop
        return loop.run_until_complete(coro)
    else:
        return loop.run_until_complete(coro)

def main():
    st.set_page_config(page_title="AI Aggregator", layout="wide")
    init_state()

    st.title("AI Aggregator — Production-ready Prototype")
    with st.sidebar:
        st.header("Settings")
        model = st.selectbox("Summarization model", ["gpt-4o-mini", "gpt-3.5-turbo"])
        relevancy_threshold = st.slider("Relevancy threshold", 0.0, 1.0, 0.6, 0.05)
        max_input_tokens = st.number_input("Max input tokens (per chunk)", value=1200, min_value=100)
        max_output_tokens = st.number_input("Max output tokens (summary)", value=256, min_value=50)

        if st.button("Clear cache & index"):
            try:
                config.CACHE_FILE.unlink(missing_ok=True)
                config.FAISS_INDEX_FILE.unlink(missing_ok=True)
                config.FAISS_MAP_FILE.unlink(missing_ok=True)
                st.success("Cleared cache and index.")
            except Exception as e:
                st.error(f"Failed to clear: {e}")

    query = st.text_input("Query", placeholder="Find cheapest electricity providers in Oslo")
    uploaded_files = st.file_uploader("Upload files", accept_multiple_files=True, type=["pdf","docx","txt","csv","json","xlsx"])
    urls_text = st.text_area("URLs (one per line)")
    urls = [u.strip() for u in urls_text.splitlines() if u.strip()]

    if st.button("Run"):
        valid = True
        if not query or (not uploaded_files and not urls):
            st.error("Enter query and at least one file or URL")
            valid = False
        if valid:
            st.session_state.query_history.append(query)
            with tempfile.TemporaryDirectory() as tmpdir:
                paths = []
                for f in uploaded_files:
                    p = Path(tmpdir) / f.name
                    p.write_bytes(f.read())
                    paths.append(str(p))

                with st.spinner("Processing..."):
                    # Use safe runner instead of asyncio.run
                    result = run_async(execute_query(query, paths, urls, model, max_input_tokens, max_output_tokens, relevancy_threshold))

                if "error" in result:
                    st.error(result["error"])
                else:
                    display_results(result["results"])
                    st.subheader("Metrics")
                    st.json(result["metrics"])

if __name__ == "__main__":
    main()
