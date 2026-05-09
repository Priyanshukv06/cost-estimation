FROM python:3.13-slim

# Set working directory
WORKDIR /app

# Copy requirements first for Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy only what's needed for inference (no training data)
COPY app/ ./app/
COPY models_new/ ./models_new/
COPY data_sample/ ./data_sample/

# Expose port
EXPOSE 8000

# Run with uvicorn (single worker for free tier memory constraints)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
