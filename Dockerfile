FROM python:3.11-slim

# System deps (psycopg2 and friends)
RUN apt-get update && apt-get install -y \
  build-essential \
  libpq-dev \
  && rm -rf /var/lib/apt/lists/*

# Workdir
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project code
COPY . .

# Ensure entrypoint is executable
RUN chmod +x /app/entrypoint.sh

# Environment defaults (override in docker-compose/env)
ENV PYTHONUNBUFFERED=1 \
  DJANGO_SETTINGS_MODULE=main_module.settings

# Use entrypoint to handle migrations/superuser
ENTRYPOINT ["/app/entrypoint.sh"]

# Default command: run Gunicorn (or change to runserver for dev)
CMD ["gunicorn", "main_module.wsgi:application", "--bind", "0.0.0.0:8000"]
