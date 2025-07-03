# Use an official lightweight Python image
FROM python:3.10-slim AS builder

# Set working directory
WORKDIR /app

# Install build dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements and install to a wheel cache
COPY requirements.txt ./
RUN pip wheel --no-cache-dir --wheel-dir /wheels -r requirements.txt

# Runtime stage
FROM python:3.10-slim
WORKDIR /app

# Copy wheels and install
COPY --from=builder /wheels /wheels
RUN pip install --no-cache /wheels/*

# Copy app
COPY . .

# Expose default Streamlit port
EXPOSE 8501

# Run the app
CMD ["streamlit", "run", "taqsense_app.py", "--server.port=8501", "--server.address=0.0.0.0"]