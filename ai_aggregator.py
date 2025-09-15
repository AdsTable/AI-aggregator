"""
AI Aggregator - Complete production-ready prototype with enhanced multi-provider support.
Enhanced for Norwegian electricity price extraction with improved data parsing.
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
                    "includeTags": ["table", "tr", "td", "th", "div", "span", "p"],
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

# -------------------- ENHANCED NORWEGIAN ELECTRICITY PRICE EXTRACTOR --------------------
def extract_norwegian_electricity_data_comprehensive(html: str) -> dict:
    """Ultra-comprehensive extractor specifically designed for Norwegian electricity sites"""
    if not html:
        return {"tables": [], "prices": [], "text_prices": [], "debug": "No HTML content"}
    
    soup = BeautifulSoup(html, "html.parser")
    all_data = []
    all_prices = []
    text_prices = []
    
    # Get full text for analysis
    full_text = soup.get_text()
    
    # Strategy 1: Norwegian provider recognition (case-insensitive)
    norwegian_providers = [
        "agva kraft", "austevoll kraft", "bodø energi", "bærum energi", "cheap energy", "dalane energi", 
        "dragefossen", "drangedal kraft", "eidefoss", "eiker energi", "eletra", "fitjar kraftlag", 
        "fjordkraft", "folkekraft", "fortum", "fosenkraft energi", "gudbrandsdal energi", 
        "hardanger energi", "haugaland kraft", "helgeland kraft strøm", "hurum kraft", "husleiestrøm", 
        "ishavskraft", "istad kraft", "jærkraft", "kilden kraft", "klarkraft", "kraftriket", 
        "kragerø kraft", "kvam kraftverk", "luster energi", "lyse", "midt energi", "motkraft", 
        "neas", "nte", "notodden energi", "polar kraft", "rauland kraft", "rauma energi", 
        "ren røros strøm", "rissa kraftlag", "saga energi", "skandiaenergi", "smart energi", 
        "sodvin energi", "strøyma kraft", "sunndal energi", "svorka energi", "telemark kraft", 
        "tibber", "tinn energi", "trøndelagskraft", "ustekveikja energi", "vokks kraft", 
        "varanger kraftmarked", "vest-telemark kraftlag", "vev romerike strøm", "vibb", 
        "viddakraft", "voss energi", "wattn", "å strøm", "hafslund", "elvia", "tensio", 
        "komplett", "nordpool", "otovo", "eidsiva", "bkk", "trønder", "vardar"
    ]
    
    found_providers = []
    for provider in norwegian_providers:
        if provider in full_text.lower():
            found_providers.append(provider)
    
    # Strategy 2: Ultra-comprehensive price pattern extraction
    price_patterns = [
        # Provider with specific prices - simplified pattern
        r'(tibber|fjordkraft|hafslund|lyse|agder|elvia|tensio|komplett|nordpool|otovo|eidsiva|bkk|trønder|vardar|fortum|polar kraft|saga energi|cheap energy|wattn|vibb)[^0-9]*?(\d+[.,]\d*)\s*(?:kr|øre)',
        
        # Monthly prices
        r'(\w+(?:\s+\w+)*)\s*[:\-]?\s*(\d+[.,]\d*)\s*kr[^0-9]*?(?:mnd|måned|per måned|månedlig|abonnement)',
        r'månedspris[:\s]*(\d+[.,]\d*)\s*kr',
        r'månedsabonnement[:\s]*(\d+[.,]\d*)\s*kr',
        r'abonnement[:\s]*(\d+[.,]\d*)\s*kr',
        
        # kWh prices  
        r'(\w+(?:\s+\w+)*)\s*[:\-]?\s*(\d+[.,]\d*)\s*øre[^0-9]*?(?:kwh|kWh)',
        r'spotpris[:\s]*(\d+[.,]\d*)\s*øre',
        r'fastpris[:\s]*(\d+[.,]\d*)\s*øre',
        r'påslag[:\s]*(\d+[.,]\d*)\s*øre',
        r'strømpris[:\s]*(\d+[.,]\d*)\s*øre',
        
        # General price patterns
        r'(\w+(?:\s+\w+)*)\s+(\d+[.,]\d*)\s*kr',
        r'(\w+(?:\s+\w+)*)\s+(\d+[.,]\d*)\s*øre',
        r'pris[:\s]*(\d+[.,]\d*)\s*(?:kr|øre)',
        
        # Table-like structured data
        r'(\w+(?:\s+\w+)*)\s+(\d+[.,]\d*)\s+(\d+[.,]\d*)',  # Provider Price1 Price2
        r'(\w+(?:\s+\w+)*)\s+(\d+[.,]\d*)\s*kr\s+(\d+[.,]\d*)\s*øre',  # Provider XX kr YY øre
        
        # Spot price specific
        r'(\w+(?:\s+\w+)*)\s*spot[^0-9]*?(\d+[.,]\d*)',
        r'(\w+(?:\s+\w+)*)\s*fast[^0-9]*?(\d+[.,]\d*)',
        
        # Numbers followed by currency
        r'(\d+[.,]\d*)\s*kr.*?(?:måned|mnd)',
        r'(\d+[.,]\d*)\s*øre.*?(?:kwh|kWh)',
        
        # More general patterns
        r'([A-ZÆØÅ][a-zæøå]+(?:\s+[A-ZÆØÅ][a-zæøå]+)*)\s*[:.]?\s*(\d+[.,]\d*)\s*kr',
        r'([A-ZÆØÅ][a-zæøå]+(?:\s+[A-ZÆØÅ][a-zæøå]+)*)\s*[:.]?\s*(\d+[.,]\d*)\s*øre',
    ]
    
    for pattern in price_patterns:
        try:
            matches = re.findall(pattern, full_text, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                if isinstance(match, tuple) and len(match) >= 1:
                    all_prices.append({
                        "type": "regex_match",
                        "data": list(match),
                        "pattern": pattern[:50] + "..."
                    })
        except re.error as e:
            logger.warning(f"Regex pattern error: {e}")
            continue
    
    # Strategy 3: Enhanced HTML table extraction
    tables = soup.find_all("table")
    for i, table in enumerate(tables):
        table_data = {"type": "html_table", "headers": [], "rows": []}
        
        # Extract headers
        header_row = table.find("tr")
        headers = []
        if header_row:
            header_cells = header_row.find_all(["th", "td"])
            if header_cells:
                headers = [cell.get_text(strip=True) for cell in header_cells]
        
        table_data["headers"] = headers
        
        # Extract all rows
        rows = []
        all_rows = table.find_all("tr")
        start_row = 1 if headers else 0
        
        for row in all_rows[start_row:]:
            cells = row.find_all(["td", "th"])
            if cells:
                row_data = [cell.get_text(strip=True) for cell in cells]
                if any(cell for cell in row_data):
                    # Check if row contains price data
                    row_text = " ".join(row_data).lower()
                    if any(word in row_text for word in ["kr", "øre", "pris", "mnd", "spot", "fast"]):
                        rows.append({"data": row_data, "contains_prices": True})
                    else:
                        rows.append({"data": row_data, "contains_prices": False})
        
        if rows:
            table_data["rows"] = rows
            # Count price-containing rows
            price_rows = sum(1 for row in rows if row.get("contains_prices", False))
            table_data["price_row_count"] = price_rows
            all_data.append(table_data)
    
    # Strategy 4: Structured sections and divs
    price_sections = soup.find_all(["div", "section", "article"], 
                                   class_=re.compile(r"price|pris|tilbud|sammenlign|compare|offer|tariff|plan", re.I))
    
    for section in price_sections:
        section_text = section.get_text()
        
        # Look for price patterns in this section
        for pattern in price_patterns[:5]:  # Use top 5 patterns
            try:
                matches = re.findall(pattern, section_text, re.IGNORECASE)
                for match in matches:
                    if isinstance(match, tuple) and len(match) >= 1:
                        all_prices.append({
                            "type": "section_match",
                            "data": list(match),
                            "section_class": section.get("class", [])
                        })
            except re.error:
                continue
    
    # Strategy 5: Text-based price extraction (line by line)
    lines = full_text.split('\n')
    for line in lines:
        line = line.strip()
        if len(line) < 5 or len(line) > 200:  # Skip very short or very long lines
            continue
        
        # Look for price mentions
        if any(word in line.lower() for word in ["kr", "øre", "pris"]):
            # Extract any numbers followed by kr or øre
            price_matches = re.findall(r'(\d+[.,]\d*)\s*(?:kr|øre)', line, re.I)
            if price_matches:
                # Try to find provider name in the same line
                provider_match = None
                for provider in norwegian_providers:
                    if provider in line.lower():
                        provider_match = provider
                        break
                
                if not provider_match:
                    # Try to extract any capitalized word as potential provider
                    words = line.split()
                    for word in words:
                        if word and word[0].isupper() and len(word) > 3:
                            provider_match = word
                            break
                
                text_prices.append({
                    "type": "text_line",
                    "line": line,
                    "prices": price_matches,
                    "provider": provider_match
                })
    
    # Strategy 6: JSON-LD structured data
    scripts = soup.find_all("script", type="application/ld+json")
    structured_data = []
    for script in scripts:
        try:
            if script.string:
                data = json.loads(script.string)
                if isinstance(data, dict) and ("offers" in data or "priceRange" in data or "price" in data):
                    structured_data.append(data)
        except:
            continue
    
    return {
        "tables": all_data,
        "prices": all_prices,
        "text_prices": text_prices,
        "found_providers": found_providers,
        "structured_data": structured_data,
        "debug": f"Found {len(all_data)} tables, {len(all_prices)} price patterns, {len(text_prices)} text prices, {len(found_providers)} providers, {len(structured_data)} structured data"
    }

# -------------------- AI PROCESSING (SYNC) --------------------
def run_async(coro):
    """Helper to run async functions in sync context"""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            return loop.run_until_complete(coro)
        else:
            return asyncio.run(coro)
    except RuntimeError:
        return asyncio.run(coro)

def extract_structured_data_ai_enhanced(query: str, html: str, provider: AIProvider) -> list:
    """Enhanced AI extraction with comprehensive data preprocessing"""
    
    # Use comprehensive Norwegian electricity extractor
    extracted = extract_norwegian_electricity_data_comprehensive(html)
    tables = extracted["tables"]
    prices = extracted["prices"]
    text_prices = extracted["text_prices"]
    found_providers = extracted["found_providers"]
    structured_data = extracted["structured_data"]
    
    # Create sample of raw text focusing on price-related content
    soup = BeautifulSoup(html, "html.parser")
    full_text = soup.get_text()
    
    # Extract price-related paragraphs
    price_paragraphs = []
    for paragraph in full_text.split('\n'):
        paragraph = paragraph.strip()
        if paragraph and any(word in paragraph.lower() for word in ["kr", "øre", "pris", "spot", "fast", "måned", "mnd", "abonnement"]):
            price_paragraphs.append(paragraph)
    
    # Enhanced prompt with more structured data
    prompt = f"""
Analyze this Norwegian electricity pricing data. Extract structured pricing information in JSON format.

QUERY: {query}

FOUND PROVIDERS: {found_providers}

EXTRACTED PRICE PATTERNS:
{json.dumps(prices[:15], ensure_ascii=False, indent=2)}

TEXT-BASED PRICES:
{json.dumps(text_prices[:10], ensure_ascii=False, indent=2)}

PRICE-RELATED PARAGRAPHS:
{chr(10).join(price_paragraphs[:20])}

EXTRACTED TABLES:
{json.dumps([{{"type": t.get("type"), "headers": t.get("headers"), "price_rows": len([r for r in t.get("rows", []) if r.get("contains_prices")])}} for t in tables], ensure_ascii=False, indent=2)}

Based on this data, extract Norwegian electricity pricing information in this EXACT JSON format:
[
  {{
    "provider": "company name (e.g., Tibber, Fjordkraft, Agva Kraft)",
    "price_monthly": "monthly price in NOK (e.g., '29 kr', '39 kr')",
    "price_kwh": "price per kWh in øre (e.g., '95.5 øre', '105 øre')",
    "påslag": "påslag per kWh in øre (e.g., '- 0.5 øre', '1 øre')",
    "contract_type": "spot/fixed/variable",
    "binding_period": "binding period (e.g., 'ingen', '12 måneder')",
    "additional_info": "any additional conditions or notes"
  }}
]

IMPORTANT INSTRUCTIONS:
1. Look for ANY price information, even if incomplete
2. Extract provider names from the found_providers list or price patterns
3. If you find monthly prices (kr/mnd), extract them
4. If you find kWh prices (øre/kWh), extract them
5. Even partial information is valuable
6. If you find pricing data, return it - don't be too strict

If you cannot find ANY pricing information at all, return:
[{{"error": "No electricity pricing data found", "analysis": "describe what you found instead"}}]
"""
    
    result_text = run_async(provider.chat(prompt))
    
    try:
        # Clean up the response text
        result_text = result_text.strip()
        
        # Try to find JSON in the response
        json_matches = re.findall(r'\[.*?\]', result_text, re.DOTALL)
        if json_matches:
            for json_match in json_matches:
                try:
                    structured_json = json.loads(json_match)
                    if isinstance(structured_json, list) and structured_json:
                        return structured_json
                except:
                    continue
        
        # Try to parse the entire response as JSON
        structured_json = json.loads(result_text)
        if isinstance(structured_json, list):
            return structured_json
        else:
            return [structured_json]
            
    except json.JSONDecodeError:
        # Enhanced fallback: try to create structured data from our extractions
        logger.error(f"JSON parsing failed for response: {result_text[:500]}")
        
        # Fall back to our own extractions
        fallback_results = []
        
        # Process price patterns
        processed_providers = set()
        for price_data in prices[:15]:
            if price_data.get("type") == "regex_match" and len(price_data.get("data", [])) >= 2:
                provider = price_data["data"][0].strip().title()
                price_info = " ".join(price_data["data"][1:])
                
                if provider not in processed_providers:
                    processed_providers.add(provider)
                    fallback_results.append({
                        "provider": provider,
                        "price_info": price_info,
                        "source": "pattern_extraction",
                        "raw_data": price_data["data"]
                    })
        
        # Process text prices
        for text_price in text_prices[:10]:
            provider = text_price.get("provider")
            if provider and provider not in processed_providers:
                processed_providers.add(provider)
                fallback_results.append({
                    "provider": provider.title(),
                    "prices_found": text_price.get("prices", []),
                    "source": "text_extraction",
                    "line": text_price.get("line", "")[:100]
                })
        
        if fallback_results:
            return fallback_results
        
        return [{
            "error": "AI response parsing failed", 
            "raw_response": result_text[:1000], 
            "extraction_debug": extracted["debug"],
            "fallback_data": {
                "providers_found": found_providers,
                "price_patterns_count": len(prices),
                "text_prices_count": len(text_prices)
            }
        }]

# -------------------- FAISS MANAGER --------------------
class FAISSManager:
    def __init__(self, dim=config.FAISS_DIM, index_path=config.FAISS_INDEX_PATH):
        self.dim = dim
        self.index_path = index_path
        try:
            self.model = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception as e:
            logger.error(f"Failed to load SentenceTransformer: {e}")
            self.model = None
        
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
        if not self.model:
            return
            
        for doc in documents:
            text = json.dumps(doc, ensure_ascii=False)
            try:
                vec = self.model.encode([text])
                self.index.add(np.array(vec, dtype='float32'))
                self.id_map[str(len(self.id_map))] = text
            except Exception as e:
                logger.error(f"Error adding document to FAISS: {e}")
        
        self.save_index()
        self.save_id_map()
    
    def save_index(self):
        try:
            faiss.write_index(self.index, str(self.index_path))
        except Exception as e:
            logger.error(f"Error saving FAISS index: {e}")
    
    def search(self, query: str, top_k=5):
        if self.index.ntotal == 0 or not self.model:
            return []
        
        try:
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
        except Exception as e:
            logger.error(f"Error searching FAISS: {e}")
            return []

# Initialize managers
try:
    faiss_manager = FAISSManager()
except Exception as e:
    logger.error(f"Error initializing FAISS manager: {e}")
    faiss_manager = None

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
    st.set_page_config(page_title="AI Aggregator - Norwegian Electricity", page_icon="⚡", layout="wide")
    st.title("⚡ AI Aggregator - Enhanced Norwegian Electricity Price Analyzer")
    
    # Sidebar with API status
    st.sidebar.header("🔑 API Key Status")
    available_providers = check_provider_availability()
    
    for name, available in available_providers.items():
        status = "✅ Available" if available else "❌ No API Key"
        st.sidebar.write(f"{name}: {status}")
    
    # Main tabs
    tab1, tab2 = st.tabs(["⚡ Enhanced Power Prices", "📊 FAISS Search"])
    
    with tab1:
        st.header("⚡ Enhanced Norwegian Power Price Analyzer")
        
        # Predefined Norwegian electricity websites
        default_urls = [
            "https://www.bytt.no/strom/strompriser/oslo",
            "https://www.fjordkraft.no/privat/strom",
            "https://tibber.com/no/priser",
            "https://www.forbrukerradet.no/strompris/",
            "https://www.lyse.no/strom/stromavtaler",
            "https://www.agderenergi.no/privat/strom/stromavtaler",
            "https://www.komplett.no/category/stromavtaler"
        ]
        
        selected_default = st.selectbox("Select a Norwegian electricity website:", 
                                      ["Custom URL"] + default_urls)
        
        if selected_default == "Custom URL":
            custom_url = st.text_input("Electricity prices URL:", 
                                     "https://www.bytt.no/strom/strompriser/oslo")
        else:
            custom_url = selected_default
            st.info(f"Using: {custom_url}")
        
        provider_name = st.selectbox("AI Provider for analysis:", list(available_providers.keys()))
        
        # Debug options
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            show_extraction_debug = st.checkbox("Show extraction debug")
        with col2:
            show_raw_text = st.checkbox("Show raw text sample")
        with col3:
            show_patterns = st.checkbox("Show found patterns") 
        with col4:
            show_comprehensive_data = st.checkbox("Show comprehensive data")
        
        if st.button("⚡ Analyze Norwegian Power Prices (Enhanced)"):
            if not available_providers.get(provider_name, False):
                st.error(f"{provider_name} API key missing!")
            else:
                with st.spinner("Analyzing Norwegian power prices with enhanced extraction..."):
                    provider = get_provider_by_name(provider_name)
                    st.info(f"Using AI Provider: {provider_name}")
                    
                    # Fetch content
                    html = fetch_page_multi_method(custom_url)
                    
                    if html:
                        st.success(f"✅ Fetched {len(html)} characters")
                        
                        # Enhanced Norwegian extraction
                        extracted = extract_norwegian_electricity_data_comprehensive(html)
                        
                        # Show debug information
                        if show_extraction_debug:
                            st.subheader("🔍 Enhanced Extraction Debug")
                            st.write(extracted.get("debug"))
                            st.write(f"Found providers: {extracted.get('found_providers')}")
                        
                        if show_raw_text:
                            st.subheader("📝 Raw Text Sample")
                            soup = BeautifulSoup(html, "html.parser")
                            text_content = soup.get_text()
                            st.text(text_content[:2000])
                        
                        if show_patterns:
                            st.subheader("🔍 Found Price Patterns")
                            prices = extracted.get("prices", [])
                            if prices:
                                for i, pattern in enumerate(prices[:15]):
                                    st.write(f"Pattern {i+1}: {pattern}")
                            else:
                                st.write("No price patterns found")
                        
                        if show_comprehensive_data:
                            st.subheader("📊 Comprehensive Extraction Data")
                            
                            # Text prices
                            text_prices = extracted.get("text_prices", [])
                            if text_prices:
                                st.write("**Text-based prices:**")
                                for i, tp in enumerate(text_prices[:10]):
                                    st.write(f"{i+1}. {tp}")
                            
                            # Structured data
                            structured_data = extracted.get("structured_data", [])
                            if structured_data:
                                st.write("**Structured data:**")
                                st.json(structured_data)
                        
                        # Show extracted tables
                        tables = extracted.get("tables", [])
                        if tables:
                            st.subheader("📊 Extracted Tables")
                            for i, table in enumerate(tables):
                                price_rows = table.get("price_row_count", 0)
                                with st.expander(f"Table {i+1} - {len(table.get('rows', []))} rows ({price_rows} with prices)"):
                                    if table.get("headers"):
                                        st.write("Headers:", table["headers"])
                                    # Show only price-containing rows
                                    price_rows_data = [row for row in table.get("rows", []) if row.get("contains_prices")]
                                    if price_rows_data:
                                        st.write("Price rows:")
                                        st.json(price_rows_data[:5])
                                    else:
                                        st.write("All rows:")
                                        st.json(table['rows'][:5])
                        
                        # Enhanced AI analysis
                        specific_query = "Extract Norwegian electricity prices for residential customers, including monthly fees and kWh prices"
                        results = extract_structured_data_ai_enhanced(specific_query, html, provider)
                        
                        st.subheader("⚡ Enhanced Structured Power Price Results")
                        st.json(results)
                        
                        # Add to FAISS if valid results
                        valid_results = [r for r in results if not r.get('error')]
                        if valid_results and faiss_manager:
                            faiss_manager.add_documents(valid_results)
                            st.success(f"Added {len(valid_results)} results to FAISS index!")
                        elif not valid_results:
                            # Show what we actually extracted
                            st.warning("No valid structured results from AI, but showing raw extraction data:")
                            
                            extraction_summary = {
                                "providers_found": extracted.get('found_providers'),
                                "price_patterns_count": len(extracted.get('prices', [])),
                                "text_prices_count": len(extracted.get('text_prices', [])),
                                "tables_with_prices": len([t for t in extracted.get('tables', []) if t.get('price_row_count', 0) > 0])
                            }
                            st.json(extraction_summary)
                            
                            # Try to add extraction summary to FAISS
                            if faiss_manager and extraction_summary.get('providers_found'):
                                faiss_manager.add_documents([extraction_summary])
                                st.info("Added extraction summary to FAISS index")
                    else:
                        st.error("❌ Could not fetch content from URL")
    
    with tab2:
        st.header("📊 FAISS Vector Search")
        
        if not faiss_manager:
            st.error("FAISS manager not available. Check SentenceTransformer installation.")
            return
        
        search_query = st.text_input("Search in FAISS index:", "cheap electricity Oslo spot price monthly fee")
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
        if faiss_manager:
            st.write(f"Total vectors: {faiss_manager.index.ntotal}")
            st.write(f"Index dimension: {faiss_manager.dim}")
        else:
            st.write("FAISS not available")

if __name__ == "__main__":
    main()