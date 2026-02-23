FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Default: run ingest (override with docker run ... or compose command)
CMD ["python", "main.py", "ingest", "--input-dir", "/data", "--pattern", "**/*", "--env", "dev", "--target", "atlas"]
