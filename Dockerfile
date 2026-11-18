FROM python:3.13-slim

# PYTHONUNBUFFERED keeps the uvicorn request log flowing to the platform's log
# viewer in real time; without it, output can sit in a buffer and look hung.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Dependencies first: this layer is cached and only rebuilt when the pins change.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Only the service code and the artifact are needed at runtime. The raw data and
# the notebook are excluded by .dockerignore and are not copied.
COPY src ./src
COPY models ./models

# Run unprivileged. Nothing in this image needs to write to disk.
RUN useradd --create-home --uid 10001 appuser
USER appuser

EXPOSE 8000

# Hosts such as Render inject PORT and expect the process to bind it. Falling back
# to 8000 keeps `docker run -p 8000:8000` working unchanged. No --reload: that is
# a development-only flag.
CMD ["sh", "-c", "uvicorn src.app:app --host 0.0.0.0 --port ${PORT:-8000}"]
