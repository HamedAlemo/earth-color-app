# Use a lightweight Python image
FROM python:3.11-slim

# Set work directory
WORKDIR /app

# Install system dependencies (needed for rasterio, GDAL)
RUN apt-get update && apt-get install -y \
    gdal-bin libgdal-dev build-essential && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Expose port
EXPOSE 8000

# Start the app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]