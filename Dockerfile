# FROM python:3.11-slim

# LABEL maintainer="Digital Reputation Crisis Manager"
# LABEL description="OpenEnv-compliant PR Crisis Simulation Environment"
# LABEL version="1.0.0"

# # Set working directory
# WORKDIR /app

# # Install system dependencies
# RUN apt-get update && apt-get install -y \
#     gcc \
#     && rm -rf /var/lib/apt/lists/*

# # Copy requirements first (layer caching)
# COPY requirements.txt .

# # Install Python dependencies
# RUN pip install --no-cache-dir -r requirements.txt

# # Copy backend source
# COPY backend/ ./backend/
# COPY inference.py .

# # Set environment variables
# ENV PYTHONPATH=/app
# ENV PYTHONUNBUFFERED=1
# ENV PORT=7860

# # OpenEnv environment variables (override at runtime)
# ENV API_BASE_URL="https://api.openai.com/v1"
# ENV MODEL_NAME="gpt-4o-mini"
# ENV OPENAI_API_KEY=""

# # Expose port
# EXPOSE 7860

# # Health check
# HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
#     CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')" || exit 1

# # Run FastAPI server
# CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]

FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y gcc nodejs npm \
    && rm -rf /var/lib/apt/lists/*

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

# Build frontend — fail loudly if it breaks
WORKDIR /app/frontend
RUN npm install && npm run build

# Verify the build output exists (catches silent failures)
RUN test -d /app/frontend/dist/assets || (echo "ERROR: Frontend build failed" && exit 1)

WORKDIR /app

# Hugging Face Spaces requires port 7860
ENV PORT=7860
EXPOSE 7860

CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT} --log-level debug"]