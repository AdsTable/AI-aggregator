# 🔎 AI Aggregator Optimized

A production-ready **AI-powered aggregator** for documents and web resources.
It fetches, parses, embeds, and summarizes data using OpenAI, FAISS, and Streamlit.

---

## 🚀 Features
- Supports **CSV, JSON, PDF, DOCX, XLSX** files
- Fetches **web pages (HTML)**
- Uses **FAISS** for efficient similarity search
- **Chunking + lazy summarization** to minimize API costs
- **Cache system** to avoid repeated calls
- **Token usage & cost tracker**
- Runs in **Docker** or locally with **Streamlit**

---

## ⚙️ Installation (Local, VS Code / Windows)

1. Clone repo
   ```bash
   git clone https://github.com/your/repo.git
   cd ai_aggregator

cp .env.example .env
# edit .env and add your OPENAI_API_KEY

pip install -r requirements.txt

streamlit run ai_aggregator.py --server.headless=True

## 🚀 Запуск тестов

pytest -v


# for Linux 
make install

make local

App will be available at: http://localhost:8501

---

## 🐳 Run with Docker

make build
make run
 or
make up

## 🛠 Development

## Format code:
make format

## Lint code:
make lint

## Logs:
make logs

## 🚀 Запуск тестов
make test
