FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# Environment variable placeholders
ENV API_BASE_URL=""
ENV API_KEY=""
ENV MODEL_NAME="gemini-2.5-flash"

# Expose port (Documentation only)
EXPOSE 7860

# Run the Flask server
CMD ["python", "app.py"]
