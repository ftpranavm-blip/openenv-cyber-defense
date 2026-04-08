FROM python:3.11-slim

WORKDIR /app

# Install minimal dependencies
RUN pip install --no-cache-dir openai

# Copy project files
COPY env/ ./env/
COPY inference.py .
COPY openenv.yaml .

# Environment variable defaults (override at runtime)
ENV API_BASE_URL="https://api.anthropic.com/v1"
ENV MODEL_NAME="claude-3-haiku-20240307"
ENV HF_TOKEN=""

CMD ["python", "inference.py"]
