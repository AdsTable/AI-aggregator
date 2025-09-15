"""
AI Aggregator - Complete production-ready prototype with enhanced multi-provider support.

Features:
 - Multi-AI provider support (OpenAI, Gemini, DeepSeek, Claude, Groq) 
 - Enhanced table extraction and fuzzy search
 - FAISS vector search with caching
 - Multiple web scraping methods
 - Advanced company search and review system
 - Streamlit UI with comprehensive provider selection
"""

import os
import json
import time
import hashlib
import logging
import threading
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import deque
from urllib.parse import urlparse
import re
import difflib

# Third-party libs
import aiohttp
import streamlit as st
import pandas as pd
import numpy as np
import faiss
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import requests
import openai
from tenacity import retry, stop_after_attempt, wait_exponential

# nest_asyncio for Streamlit compatibility
import nest_asyncio
nest_asyncio.apply()

load_dotenv()

# -------------------- Configuration --------------------
@dataclass
class Config:
    # Search API Keys
    BING_API_KEY: str = os.getenv("BING_API_KEY", "")
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
    GOOGLE_CX: str = os.getenv("GOOGLE_CX", "")
    FIRECRAWL_API_KEY: str = os.getenv("FIRECRAWL_API_KEY", "")
    
    # AI Provider API Keys
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    DEEPSEEK_API_KEY: str = os.getenv("DEEPSEEK_API_KEY", "")
    CLAUDE_API_KEY: str = os.getenv("CLAUDE_API_KEY", "")
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    
    # Configuration
    MAX_RESULTS: int = int(os.getenv("MAX_RESULTS", 10))
    CACHE_DIR: Path = Path(os.getenv("CACHE_DIR", "./cache"))
    CACHE_TTL_HOURS: int = int(os.getenv("CACHE_TTL_HOURS", 24))
    FAISS_INDEX_PATH: Path = Path(os.getenv("FAISS_INDEX_PATH", "./faiss.index"))
    FAISS_DIM: int = 384
    CHUNK_SIZE: int = 900
    CHUNK_OVERLAP: int = 100
    MAX_CONCURRENT_REQUESTS: int = 5
    CONNECTION_TIMEOUT: int = 30

config = Config()
config.CACHE_DIR.mkdir(exist_ok=True)

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ai_aggregator")

# -------------------- AI PROVIDER CLASSES --------------------
class AIProvider:
    async def chat(self, prompt: str) -> str:
        raise NotImplementedError

class OpenAIProvider(AIProvider):
    def __init__(self, api_key, model="gpt-3.5-turbo"):
        self.client = openai.Client(api_key=api_key)
        self.model = model

    async def chat(self, prompt: str) -> str:
        def sync_call():
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=1500,
                    temperature=0,
                )
                return response.choices[0].message.content.strip()
            except openai.RateLimitError:
                return "[OpenAI quota exceeded] " + prompt[:200]
            except Exception as e:
                return f"[OpenAI error: {e}] {prompt[:200]}"
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, sync_call)

class GeminiProvider(AIProvider):
    def __init__(self, api_key):
        self.api_key = api_key
        self.model = "gemini-pro"

    async def chat(self, prompt: str) -> str:
        def sync_call():
            try:
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
                headers = {"Content-Type": "application/json"}
                params = {"key": self.api_key}
                payload = {
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {"temperature": 0, "maxOutputTokens": 1500}
                }
                response = requests.post(url, headers=headers, params=params, json=payload, timeout=30)
                data = response.json()
                if "candidates" in data and data["candidates"]:
                    return data["candidates"][0]["content"]["parts"][0]["text"]
                return f"[Gemini error: {data}]"
            except Exception as e:
                return f"[Gemini error: {e}] {prompt[:200]}"
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, sync_call)

class DeepSeekProvider(AIProvider):
    def __init__(self, api_key):
        self.api_key = api_key
        self.model = "deepseek-chat"

    async def chat(self, prompt: str) -> str:
        def sync_call():
            try:
                url = "https://api.deepseek.com/v1/chat/completions"
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
                payload = {
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 1500,
                    "temperature": 0
                }
                response = requests.post(url, headers=headers, json=payload, timeout=30)
                data = response.json()
                return data["choices"][0]["message"]["content"].strip()
            except Exception as e:
                return f"[DeepSeek error: {e}] {prompt[:200]}"
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, sync_call)

class ClaudeProvider(AIProvider):
    def __init__(self, api_key):
        self.api_key = api_key
        self.model = "claude-3-haiku-20240307"

    async def chat(self, prompt: str) -> str:
        def sync_call():
            try:
                url = "https://api.anthropic.com/v1/messages"
                headers = {
                    "x-api-key": self.api_key,
                    "Content-Type": "application/json",
                    "anthropic-version": "2023-06-01"
                }
                payload = {
                    "model": self.model,
                    "max_tokens": 1500,
                    "messages": [{"role": "user", "content": prompt}]
                }
                response = requests.post(url, headers=headers, json=payload, timeout=30)
                data = response.json()
                return data["content"][0]["text"]
            except Exception as e:
                return f"[Claude error: {e}] {prompt[:200]}"
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, sync_call)

class GroqProvider(AIProvider):
    def __init__(self, api_key):
        self.api_key = api_key
        self.model = "llama3-8b-8192"

    async def chat(self, prompt: str) -> str:
        def sync_call():
            try:
                url = "https://api.groq.com/openai/v1/chat/completions"
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
                payload = {
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 1500,
                    "temperature": 0
                }
                response = requests.post(url, headers=headers, json=payload, timeout=30)
                data = response.json()
                
                if "choices" in data and data["choices"]:
                    return data["choices"][0]["message"]["content"].strip()
                elif "error" in data:
                    return f"[Groq API error: {data['error']}]"
                else:
                    return f"[Groq unexpected response: {data}]"
                    
            except Exception as e:
                return f"[Groq error: {e}] {prompt[:200]}"
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, sync_call)

class MockProviderBase(AIProvider):
    def __init__(self, name: str):
        self.name = name

    async def chat(self, prompt: str) -> str:
        return f"[{self.name} mock - no API key] {prompt[:200]}"

# -------------------- PROVIDER FACTORY --------------------
def get_provider_by_name(name: str) -> AIProvider:
    name = name.lower()
    if name == "openai" and config.OPENAI_API_KEY:
        return OpenAIProvider(config.OPENAI_API_KEY)
    if name == "gemini" and config.GEMINI_API_KEY:
        return GeminiProvider(config.GEMINI_API_KEY)
    if name == "deepseek" and config.DEEPSEEK_API_KEY:
        return DeepSeekProvider(config.DEEPSEEK_API_KEY)
    if name == "claude" and config.CLAUDE_API_KEY:
        return ClaudeProvider(config.CLAUDE_API_KEY)
    if name == "groq" and config.GROQ_API_KEY:
        return GroqProvider(config.GROQ_API_KEY)
    return MockProviderBase(f"{name}")

# -------------------- ENHANCED SEARCH ENGINE --------------------
class EnhancedSearchEngine:
    def __init__(self):
        self.search_index = {}
        self.company_data = []
    
    def add_company_data(self, companies: list):
        self.company_data.extend(companies)
        self._build_search_index()
    
    def _build_search_index(self):
        for company in self.company_data:
            name = company.get('name', '').lower()
            industry = company.get('industry', '').lower()
            rating = str(company.get('rating', ''))
            
            searchable_text = f"{name} {industry} {rating}"
            self.search_index[company.get('id', len(self.search_index))] = {
                'text': searchable_text,
                'company': company
            }
    
    def fuzzy_search(self, query: str, threshold: float = 0.6) -> list:
        query = query.lower()
        results = []
        
        for idx, data in self.search_index.items():
            similarity = difflib.SequenceMatcher(None, query, data['text']).ratio()
            
            if similarity >= threshold:
                result = data['company'].copy()
                result['search_score'] = similarity
                result['highlighted_name'] = self._highlight_match(
                    data['company'].get('name', ''), query
                )
                results.append(result)
        
        return sorted(results, key=lambda x: x['search_score'], reverse=True)
    
    def _highlight_match(self, text: str, query: str) -> str:
        pattern = re.compile(re.escape(query), re.IGNORECASE)
        return pattern.sub(f"**{query}**", text)
    
    def advanced_search(self, query: str, filters: dict = None) -> dict:
        results = self.fuzzy_search(query)
        
        if filters:
            if 'industry' in filters:
                results = [r for r in results if filters['industry'].lower() in r.get('industry', '').lower()]
            if 'min_rating' in filters:
                results = [r for r in results if r.get('rating', 0) >= filters['min_rating']]
        
        # Group by industry
        grouped_results = {}
        for result in results:
            industry = result.get('industry', 'Other')
            if industry not in grouped_results:
                grouped_results[industry] = []
            grouped_results[industry].append(result)
        
        return grouped_results

# -------------------- CACHE FUNCTIONS --------------------
def cache_key(query: str) -> str:
    return hashlib.sha256(query.encode("utf-8")).hexdigest()

def save_cache(key: str, data: dict):
    data["_timestamp"] = datetime.utcnow().isoformat()
    cache_file = config.CACHE_DIR / f"{key}.json"
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)

def load_cache(key: str) -> dict:
    cache_file = config.CACHE_DIR / f"{key}.json"
    if cache_file.exists():
        with open(cache_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def is_cache_expired(data: dict, ttl_hours: int = config.CACHE_TTL_HOURS) -> bool:
    timestamp = data.get("_timestamp")
    if not timestamp:
        return True
    dt = datetime.fromisoformat(timestamp)
    return datetime.utcnow() - dt > timedelta(hours=ttl_hours)

# -------------------- WEB SCRAPING FUNCTIONS (SYNC) --------------------
def fetch_direct_html(url: str) -> str:
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        return response.text
    except Exception as e:
        logger.error(f"Direct fetch error: {e}")
        return ""

def fetch_firecrawl_enhanced(url: str) -> str:
    if not config.FIRECRAWL_API_KEY:
        return ""
    
    endpoints = [
        "https://api.firecrawl.dev/v1/scrape",
        "https://api.firecrawl.dev/v0/scrape",
    ]
    
    for api_url in endpoints:
        try:
            headers = {"Authorization": f"Bearer {config.FIRECRAWL_API_KEY}"}
            payloads = [
                {
                    "url": url,
                    "formats": ["html"],
                    "includeTags": ["table", "tr", "td", "th", "div"],
                    "waitFor": 5000
                },
                {"url": url}
            ]
            
            for payload in payloads:
                try:
                    response = requests.post(api_url, headers=headers, json=payload, timeout=30)
                    if response.status_code == 200:
                        data = response.json()
                        html_paths = [
                            ["data", "html"],
                            ["html"],
                            ["content"],
                            ["data", "content"]
                        ]
                        
                        for path in html_paths:
                            content = data
                            for key in path:
                                if isinstance(content, dict) and key in content:
                                    content = content[key]
                                else:
                                    content = None
                                    break
                            
                            if content and len(str(content)) > 1000:
                                return str(content)
                except Exception:
                    continue
        except Exception:
            continue
    return ""

def fetch_page_multi_method(url: str) -> str:
    """SYNC version - no await needed"""
    methods = [
        ("Firecrawl Enhanced", fetch_firecrawl_enhanced),
        ("Direct Requests", fetch_direct_html),
    ]
    
    for method_name, method_func in methods:
        try:
            html = method_func(url)
            if html and len(html) > 1000:
                logger.info(f"✅ {method_name} succeeded: {len(html)} chars")
                return html
            else:
                logger.info(f"❌ {method_name} failed: {len(html) if html else 0} chars")
        except Exception as e:
            logger.error(f"❌ {method_name} error: {e}")
    
    return ""

# -------------------- SEARCH FUNCTIONS --------------------
def search_bing(query: str, region: str = "no") -> list:
    if not config.BING_API_KEY:
        return []
    key = cache_key(f"bing_{query}_{region}")
    cached = load_cache(key)
    if cached and not is_cache_expired(cached):
        return cached.get("urls", [])
    
    headers = {"Ocp-Apim-Subscription-Key": config.BING_API_KEY}
    params = {"q": query, "count": config.MAX_RESULTS, "mkt": region}
    try:
        response = requests.get("https://api.bing.microsoft.com/v7.0/search", headers=headers, params=params)
        data = response.json()
        links = [item["url"] for item in data.get("webPages", {}).get("value", [])]
        save_cache(key, {"urls": links})
        return links
    except Exception as e:
        logger.error(f"Bing search error: {e}")
        return []

def search_google(query: str, region: str = "no") -> list:
    if not config.GOOGLE_API_KEY or not config.GOOGLE_CX:
        return []
    key = cache_key(f"google_{query}_{region}")
    cached = load_cache(key)
    if cached and not is_cache_expired(cached):
        return cached.get("urls", [])
    
    params = {"key": config.GOOGLE_API_KEY, "cx": config.GOOGLE_CX, "q": query, "num": config.MAX_RESULTS, "gl": region}
    try:
        response = requests.get("https://www.googleapis.com/customsearch/v1", params=params)
        data = response.json()
        links = [item["link"] for item in data.get("items", [])]
        save_cache(key, {"urls": links})
        return links
    except Exception as e:
        logger.error(f"Google search error: {e}")
        return []

def multi_search(query: str, region: str = "no") -> list:
    bing_links = search_bing(query, region)
    google_links = search_google(query, region)
    combined = list(dict.fromkeys(bing_links + google_links))
    return combined[:config.MAX_RESULTS]

# -------------------- ENHANCED TABLE EXTRACTION --------------------
def extract_tables_comprehensive(html: str) -> dict:
    """Comprehensive table extraction from HTML with multiple strategies"""
    if not html:
        return {"tables": [], "debug": "No HTML content"}
    
    soup = BeautifulSoup(html, "html.parser")
    all_tables = []
    
    # Strategy 1: Traditional HTML tables
    html_tables = soup.find_all("table")
    logger.info(f"Found {len(html_tables)} HTML tables")
    
    for i, table in enumerate(html_tables):
        table_data = {"type": "html_table", "headers": [], "rows": []}
        
        # Extract headers
        headers = []
        for header_row in table.find_all("tr")[:3]:  # Check first 3 rows for headers
            header_cells = header_row.find_all(["th", "td"])
            if header_cells:
                potential_headers = [cell.get_text(strip=True) for cell in header_cells]
                # If this looks like headers (short text, no numbers)
                if all(len(h) < 50 and not re.search(r'\d+[.,]\d+', h) for h in potential_headers if h):
                    headers = potential_headers
                    break
        
        table_data["headers"] = headers
        
        # Extract rows
        rows = []
        all_rows = table.find_all("tr")
        start_row = 1 if headers else 0
        
        for row in all_rows[start_row:]:
            cells = row.find_all(["td", "th"])
            if cells:
                row_data = [cell.get_text(strip=True) for cell in cells]
                if any(cell for cell in row_data):  # At least one non-empty cell
                    if headers:
                        row_dict = dict(zip(headers, row_data))
                    else:
                        row_dict = {"data": row_data}
                    rows.append(row_dict)
        
        if rows:
            table_data["rows"] = rows
            all_tables.append(table_data)
            logger.info(f"Extracted HTML table {i+1} with {len(rows)} rows")
    
    # Strategy 2: Div-based grid structures
    grid_containers = soup.find_all("div", class_=re.compile(r"table|grid|list|data|row", re.I))
    logger.info(f"Found {len(grid_containers)} potential grid containers")
    
    for container in grid_containers:
        rows = []
        # Look for repeated div patterns
        items = container.find_all("div", class_=re.compile(r"row|item|entry|cell", re.I))
        
        if len(items) >= 3:  # At least 3 items to consider as table
            for item in items[:50]:  # Limit to 50 items
                # Extract text from various child elements
                text_elements = []
                for elem in item.find_all(["span", "div", "p", "strong", "a"]):
                    text = elem.get_text(strip=True)
                    if text and len(text) < 200:
                        text_elements.append(text)
                
                if len(text_elements) >= 2:  # At least 2 data points
                    rows.append({"data": text_elements})
            
            if len(rows) >= 3:
                all_tables.append({"type": "div_grid", "rows": rows})
                logger.info(f"Extracted div-grid with {len(rows)} rows")
    
    # Strategy 3: Look for structured data patterns (e.g., JSON-LD)
    scripts = soup.find_all("script", type="application/ld+json")
    for script in scripts:
        try:
            data = json.loads(script.string)
            if isinstance(data, dict) and any(key in data for key in ["itemListElement", "hasOfferCatalog", "offers"]):
                # Convert structured data to table format
                structured_rows = []
                if "itemListElement" in data:
                    for item in data["itemListElement"]:
                        if isinstance(item, dict):
                            structured_rows.append({"data": [str(v) for v in item.values() if v]})
                if structured_rows:
                    all_tables.append({"type": "json_ld", "rows": structured_rows})
                    logger.info(f"Extracted JSON-LD with {len(structured_rows)} items")
        except:
            continue
    
    # Strategy 4: Text pattern extraction for Norwegian electricity prices
    text_content = soup.get_text()
    price_patterns = [
        r"(\w+(?:\s+\w+)*)\s*[:\-]\s*(\d+[.,]\d*)\s*(?:kr|øre)(?:/mnd|/kWh)?",  # Provider: 123.45 kr/mnd
        r"(\w+(?:\s+\w+)*)\s+(\d+[.,]\d*)\s*kr.*?(\d+[.,]\d*)\s*øre",  # Provider 123 kr ... 45 øre
        r"Tilbyder[:\s]*(\w+(?:\s+\w+)*)\s+.*?(\d+[.,]\d*)",  # Tilbyder: Provider ... 123.45
    ]
    
    pattern_matches = []
    for pattern in price_patterns:
        matches = re.findall(pattern, text_content, re.IGNORECASE | re.MULTILINE)
        for match in matches[:20]:  # Limit matches
            if isinstance(match, tuple) and len(match) >= 2:
                pattern_matches.append({"data": list(match)})
    
    if pattern_matches:
        all_tables.append({"type": "pattern_extraction", "rows": pattern_matches})
        logger.info(f"Extracted {len(pattern_matches)} price patterns")
    
    return {
        "tables": all_tables,
        "debug": f"Found {len(all_tables)} table structures: {len(html_tables)} HTML tables, {len(grid_containers)} grids, {len(scripts)} JSON-LD, {len(pattern_matches)} patterns"
    }

# -------------------- AI PROCESSING (SYNC) --------------------
def run_async(coro):
    """Helper to run async functions in sync context (like in v1)"""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            return loop.run_until_complete(coro)
        else:
            return asyncio.run(coro)
    except RuntimeError:
        return asyncio.run(coro)

def extract_structured_data_ai(query: str, html: str, provider: AIProvider) -> list:
    """Extract structured data using AI with comprehensive table extraction (SYNC VERSION)"""
    extracted = extract_tables_comprehensive(html)
    tables = extracted["tables"]
    
    if not tables:
        # If no tables found, try direct text extraction
        soup = BeautifulSoup(html, "html.parser")
        text_content = soup.get_text()[:5000]  # First 5000 chars
        
        prompt = f"""
Analyze this webpage text for Norwegian electricity/energy price information:

Query: {query}
Text content: {text_content}

Extract structured data in JSON format:
[
  {{
    "provider": "company name",
    "price_monthly": "monthly price if available", 
    "price_kwh": "price per kWh if available",
    "contract_type": "contract type",
    "binding_period": "binding period",
    "additional_info": "any other relevant info"
  }}
]

If no structured pricing data found, return: [{{"error": "No structured pricing data found in content"}}]
"""
    else:
        # Use extracted tables
        prompt = f"""
Analyze these extracted tables for Norwegian electricity pricing information:

Query: {query}
Extracted tables: {json.dumps(tables, ensure_ascii=False)[:8000]}

Extract structured pricing data in JSON format:
[
  {{
    "provider": "provider/company name",
    "price_monthly": "monthly price in NOK", 
    "price_kwh": "price per kWh in øre",
    "contract_type": "spot/fixed price type",
    "binding_period": "binding time period",
    "consumption": "yearly consumption kWh",
    "payment_method": "billing method",
    "additional_conditions": "other conditions"
  }}
]

Focus on offers matching: 16000 kWh/year, spot prices, no binding period, post-payment billing.
If no relevant data found, return: [{{"error": "No relevant electricity pricing data found"}}]
"""
    
    result_text = run_async(provider.chat(prompt))
    
    try:
        structured_json = json.loads(result_text)
        if isinstance(structured_json, list):
            return structured_json
        else:
            return [structured_json]
    except json.JSONDecodeError:
        # Try to extract JSON from response text
        json_match = re.search(r'\[.*\]', result_text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except:
                pass
        
        return [{"error": "AI output parsing failed", "raw": result_text[:1000]}]

# -------------------- FAISS MANAGER --------------------
class FAISSManager:
    def __init__(self, dim=config.FAISS_DIM, index_path=config.FAISS_INDEX_PATH):
        self.dim = dim
        self.index_path = index_path
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        
        if self.index_path.exists():
            try:
                self.index = faiss.read_index(str(self.index_path))
                self.id_map = self.load_id_map()
            except:
                self.index = faiss.IndexFlatL2(dim)
                self.id_map = {}
        else:
            self.index = faiss.IndexFlatL2(dim)
            self.id_map = {}
    
    def load_id_map(self):
        map_path = str(self.index_path) + ".map.json"
        if os.path.exists(map_path):
            with open(map_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}
    
    def save_id_map(self):
        map_path = str(self.index_path) + ".map.json"
        with open(map_path, "w", encoding="utf-8") as f:
            json.dump(self.id_map, f, ensure_ascii=False)
    
    def add_documents(self, documents: list):
        for doc in documents:
            text = json.dumps(doc, ensure_ascii=False)
            vec = self.model.encode([text])
            self.index.add(np.array(vec, dtype='float32'))
            self.id_map[str(len(self.id_map))] = text
        
        self.save_index()
        self.save_id_map()
    
    def save_index(self):
        faiss.write_index(self.index, str(self.index_path))
    
    def search(self, query: str, top_k=5):
        if self.index.ntotal == 0:
            return []
        
        vec = self.model.encode([query])
        vec = np.array(vec, dtype='float32')
        if vec.ndim == 1:
            vec = vec.reshape(1, -1)
        
        D, I = self.index.search(vec, min(top_k, self.index.ntotal))
        results = []
        
        for dist, idx in zip(D[0], I[0]):
            if str(idx) in self.id_map:
                try:
                    doc = json.loads(self.id_map[str(idx)])
                    results.append(doc)
                except:
                    continue
        
        return results

# Initialize managers
faiss_manager = FAISSManager()
search_engine = EnhancedSearchEngine()

# -------------------- STREAMLIT UI --------------------
def check_provider_availability():
    providers = {
        "OpenAI": bool(config.OPENAI_API_KEY),
        "Gemini": bool(config.GEMINI_API_KEY),
        "DeepSeek": bool(config.DEEPSEEK_API_KEY),
        "Claude": bool(config.CLAUDE_API_KEY),
        "Groq": bool(config.GROQ_API_KEY),
    }
    return providers

def main():
    st.set_page_config(page_title="AI Aggregator Complete", page_icon="🤖", layout="wide")
    st.title("🤖 AI Aggregator - Complete Edition")
    
    # Sidebar with API status
    st.sidebar.header("🔑 API Key Status")
    available_providers = check_provider_availability()
    
    for name, available in available_providers.items():
        status = "✅ Available" if available else "❌ No API Key"
        st.sidebar.write(f"{name}: {status}")
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🔍 Smart Search", "⚡ Power Prices", "🏢 Company Search", "📊 FAISS Search"])
    
    with tab1:
        st.header("🔍 Smart Search & AI Processing")
        
        # Provider selection
        provider_options = []
        for name, available in available_providers.items():
            if available:
                provider_options.append(f"{name} ✅")
            else:
                provider_options.append(f"{name} ❌")
        
        if provider_options:
            selected_provider_display = st.selectbox("Select AI Provider:", provider_options)
            selected_provider = selected_provider_display.split(" ")[0]
        else:
            st.error("No AI providers available!")
            selected_provider = "mock"
        
        def get_selected_provider():
            if not available_providers.get(selected_provider, False):
                st.warning(f"{selected_provider} API key missing!")
                return MockProviderBase(selected_provider)
            return get_provider_by_name(selected_provider)
        
        # Search inputs
        col1, col2 = st.columns([3, 1])
        with col1:
            query = st.text_input("Search query:", "electricity prices Oslo Norway")
        with col2:
            region = st.selectbox("Region:", ["no", "se", "dk", "us", "uk"])
        
        # Search execution
        search_keys_valid = bool(config.BING_API_KEY or config.GOOGLE_API_KEY)
        
        if not search_keys_valid:
            st.error("Search API keys missing! Set BING_API_KEY or GOOGLE_API_KEY")
        
        if st.button("🔍 Search & Analyze", disabled=not search_keys_valid):
            with st.spinner("Searching and analyzing..."):
                provider = get_selected_provider()
                st.info(f"Using AI Provider: {selected_provider}")
                
                # Multi-source search
                links = multi_search(query, region)
                st.write(f"Found {len(links)} search results")
                
                if links:
                    all_results = []
                    
                    # Process top 3 links
                    for i, url in enumerate(links[:3]):
                        st.write(f"Processing {i+1}/{min(3, len(links))}: {url}")
                        
                        html = fetch_page_multi_method(url)  # SYNC CALL - NO AWAIT
                        if html:
                            st.write(f"✅ Fetched {len(html)} chars")
                            
                            # Extract structured data
                            results = extract_structured_data_ai(query, html, provider)  # SYNC CALL
                            if results:
                                # Add source URL to results
                                for result in results:
                                    if isinstance(result, dict):
                                        result['source_url'] = url
                                all_results.extend(results)
                        else:
                            st.write("❌ Failed to fetch content")
                    
                    if all_results:
                        st.subheader("📊 Extracted Data")
                        
                        # Filter out errors for display
                        valid_results = [r for r in all_results if not r.get('error')]
                        error_results = [r for r in all_results if r.get('error')]
                        
                        if valid_results:
                            st.json(valid_results)
                            
                            # Add to FAISS
                            faiss_manager.add_documents(valid_results)
                            st.success(f"Added {len(valid_results)} results to FAISS index!")
                        
                        if error_results:
                            st.subheader("⚠️ Processing Errors")
                            st.json(error_results)
                    else:
                        st.warning("No structured data extracted")
                else:
                    st.warning("No search results found")
    
    with tab2:
        st.header("⚡ Norwegian Power Prices")
        
        custom_url = st.text_input(
            "Electricity prices URL:", 
            "https://www.bytt.no/strom/strompriser/oslo"
        )
        
        provider_name = st.selectbox("AI Provider for analysis:", list(available_providers.keys()))
        
        if st.button("⚡ Analyze Power Prices"):
            if not available_providers.get(provider_name, False):
                st.error(f"{provider_name} API key missing!")
            else:
                with st.spinner("Analyzing power prices..."):
                    provider = get_provider_by_name(provider_name)
                    st.info(f"Using AI Provider: {provider_name}")
                    
                    # Fetch content
                    html = fetch_page_multi_method(custom_url)  # SYNC CALL - NO AWAIT
                    
                    if html:
                        st.success(f"✅ Fetched {len(html)} characters")
                        
                        # Show extraction debug info
                        if st.checkbox("Show debug info"):
                            extracted = extract_tables_comprehensive(html)
                            st.write("Extraction debug:", extracted.get("debug"))
                            
                            if extracted["tables"]:
                                st.subheader("📊 Extracted Tables")
                                for i, table in enumerate(extracted["tables"]):
                                    with st.expander(f"Table {i+1} ({table.get('type')})"):
                                        if 'rows' in table:
                                            st.json(table['rows'][:5])
                        
                        # AI analysis
                        specific_query = "Extract Norwegian electricity prices for 16000 kWh yearly consumption, spot prices, no binding period, post-payment billing"
                        results = extract_structured_data_ai(specific_query, html, provider)  # SYNC CALL
                        
                        st.subheader("⚡ Structured Power Price Results")
                        st.json(results)
                        
                        # Add to FAISS if valid results
                        valid_results = [r for r in results if not r.get('error')]
                        if valid_results:
                            faiss_manager.add_documents(valid_results)
                            st.success("Results added to FAISS index!")
                    else:
                        st.error("❌ Could not fetch content from URL")
    
    with tab3:
        st.header("🏢 Company Search & Reviews")
        
        # Sample companies
        sample_companies = [
            {"id": 1, "name": "Fjordkraft", "industry": "Energy", "rating": 4.2},
            {"id": 2, "name": "Tibber", "industry": "Energy", "rating": 4.5},
            {"id": 3, "name": "Lyse Energi", "industry": "Energy", "rating": 4.0},
            {"id": 4, "name": "Hafslund Strøm", "industry": "Energy", "rating": 3.8},
            {"id": 5, "name": "TechCorp AS", "industry": "Technology", "rating": 4.3},
        ]
        
        search_engine.add_company_data(sample_companies)
        
        # Search interface
        search_query = st.text_input("🔍 Search companies:", "")
        
        col1, col2 = st.columns(2)
        with col1:
            industry_filter = st.text_input("Industry filter:", "")
        with col2:
            min_rating = st.slider("Minimum rating:", 0.0, 5.0, 0.0)
        
        if search_query:
            filters = {}
            if industry_filter:
                filters['industry'] = industry_filter
            if min_rating > 0:
                filters['min_rating'] = min_rating
            
            results = search_engine.advanced_search(search_query, filters)
            
            if results:
                st.subheader("🏢 Search Results")
                for industry, companies in results.items():
                    st.write(f"**{industry}** ({len(companies)} companies)")
                    
                    for company in companies:
                        col1, col2, col3 = st.columns([3, 1, 1])
                        
                        with col1:
                            st.markdown(f"**{company.get('highlighted_name', company.get('name'))}**")
                        with col2:
                            st.metric("Rating", f"{company.get('rating', 0):.1f}/5")
                        with col3:
                            st.metric("Match", f"{company.get('search_score', 0):.0%}")
                        
                        st.divider()
    
    with tab4:
        st.header("📊 FAISS Vector Search")
        
        search_query = st.text_input("Search in FAISS index:", "cheap electricity Oslo")
        top_k = st.slider("Number of results:", 1, 20, 5)
        
        if st.button("🔍 Search FAISS"):
            if search_query:
                results = faiss_manager.search(search_query, top_k)
                
                if results:
                    st.subheader(f"📊 Found {len(results)} results")
                    for i, result in enumerate(results):
                        with st.expander(f"Result {i+1}"):
                            st.json(result)
                else:
                    st.info("No results found in FAISS index")
            else:
                st.warning("Please enter a search query")
        
        # FAISS statistics
        st.subheader("📈 FAISS Index Statistics")
        st.write(f"Total vectors: {faiss_manager.index.ntotal}")
        st.write(f"Index dimension: {faiss_manager.dim}")

if __name__ == "__main__":
    main()