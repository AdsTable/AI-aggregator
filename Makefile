PROJECT_NAME=ai-aggregator
IMAGE_NAME=$(PROJECT_NAME):latest
CONTAINER_NAME=$(PROJECT_NAME)

# -----------------------------
# Environment check
# -----------------------------
check-env:
	@if [ ! -f .env ]; then \
		echo "❌ ERROR: .env file not found!"; \
		echo "👉 Run: make init"; \
		exit 1; \
	fi
	@if ! grep -q "OPENAI_API_KEY=" .env; then \
		echo "❌ ERROR: OPENAI_API_KEY is missing in .env!"; \
		exit 1; \
	fi
	@echo "✅ Environment check passed!"

# -----------------------------
# Initialization
# -----------------------------
init:
	@if [ ! -f .env ]; then \
		cp .env.example .env; \
		echo "⚠️  Created .env (edit it and add your OPENAI_API_KEY)"; \
	fi
	@mkdir -p cache logs
	@echo "✅ Project initialized."

# -----------------------------
# Docker commands
# -----------------------------
build: check-env
	docker build -t $(IMAGE_NAME) .

run: check-env
	docker run -it --rm \
		--name $(CONTAINER_NAME) \
		-p 8501:8501 \
		--env-file .env \
		-v $(PWD)/cache:/app/cache \
		-v $(PWD)/logs:/app/logs \
		$(IMAGE_NAME)

up: check-env
	docker-compose up --build

stop:
	docker-compose down

clean:
	docker rm -f $(CONTAINER_NAME) || true
	docker rmi $(IMAGE_NAME) || true

# -----------------------------
# Local development
# -----------------------------
install: check-env
	pip install -r requirements.txt

local: check-env
	streamlit run ai-aggregator-fixed.py

format:
	black .

lint:
	flake8 .

logs:
	docker logs -f $(CONTAINER_NAME)

help:
	@echo "Available commands:"
	@echo " make init      - Initialize project (create .env, dirs)"
	@echo " make build     - Build Docker image"
	@echo " make run       - Run container"
	@echo " make up        - Run with docker-compose"
	@echo " make stop      - Stop docker-compose"
	@echo " make clean     - Remove container & image"
	@echo " make install   - Install deps locally"
	@echo " make local     - Run Streamlit locally"
	@echo " make format    - Format code"
	@echo " make lint      - Lint code"
	@echo " make logs      - Show container logs"
