FROM python:3.9-slim

WORKDIR /app

# Install build tools
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application
COPY spectre_detector.py .

# Expose the port
EXPOSE 5000

# Run the app
CMD ["python", "spectre_detector.py"]