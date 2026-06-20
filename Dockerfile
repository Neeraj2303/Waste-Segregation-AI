FROM python:3.10-slim

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /code

# Copy and install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir gunicorn

# Copy all application files
COPY . .

# Expose port 7860 (Hugging Face default)
EXPOSE 7860

# Run using Gunicorn reverse proxy
CMD ["gunicorn", "-b", "0.0.0.0:7860", "--timeout", "120", "app:app"]
