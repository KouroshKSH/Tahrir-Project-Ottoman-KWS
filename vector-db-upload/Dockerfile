FROM python:3.10-slim

# Increase pip’s default socket timeout and retry attempts
ENV PIP_DEFAULT_TIMEOUT=100 \
    PIP_MAX_RETRIES=10

# Install any apt deps your project needs
RUN apt-get update && apt-get install -y \
    libglib2.0-0 libjpeg-dev && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only requirements first so we can cache this layer
COPY requirements.txt .

# Upgrade pip, then install with higher timeout & retries
RUN pip install --upgrade pip \
 && pip install --no-cache-dir \
      --default-timeout=1000 \
      -r requirements.txt

# Now copy the rest of your code and model
COPY preprocess.py main.py ./
COPY models/siamese_model_val_arak.pth models/siamese_model_val_arak.pth

EXPOSE 8080
CMD ["uvicorn","main:app","--host","0.0.0.0","--port","8080"]

