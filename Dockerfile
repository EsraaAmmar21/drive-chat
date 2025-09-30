FROM python:3.11-slim

# System deps for PyMuPDF (fitz)
RUN apt-get update && apt-get install -y \
    build-essential gcc \
    libglib2.0-0 libgl1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy app
COPY . /app

# Hugging Face Spaces expects the app to listen on port 7860
ENV PORT=7860
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"]
