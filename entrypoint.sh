#!/bin/sh
set -e

# Optional: print where we are
echo "Current dir: $(pwd)"

ARTIFACT_DIR="${ARTIFACT_DIR:-/artifacts}"

mkdir -p "$ARTIFACT_DIR/sl" \
         "$ARTIFACT_DIR/il" \
         "$ARTIFACT_DIR/eval" \
         "$ARTIFACT_DIR/sl/tmp"
         
# Wait for Postgres (simple retry loop)
echo "Waiting for database to be ready..."
RETRIES=1
until python manage.py migrate --check >/dev/null 2>&1 || [ "$RETRIES" -eq 0 ]; do
  echo "DB not ready yet, retrying... ($RETRIES left)"
  RETRIES=$((RETRIES - 1))
  sleep 3
done

# Run migrations (idempotent)
echo "Applying database migrations..."
python manage.py migrate --noinput

# Create Django superuser if not exists
if [ -n "$DJANGO_SUPERUSER_USERNAME" ] && [ -n "$DJANGO_SUPERUSER_EMAIL" ]; then
  echo "Ensuring Django superuser exists..."
  python manage.py createsuperuser \
    --noinput \
    --username "$DJANGO_SUPERUSER_USERNAME" \
    --email "$DJANGO_SUPERUSER_EMAIL" \
    || echo "Superuser already exists or creation failed (likely already created)."
else
  echo "DJANGO_SUPERUSER_USERNAME or DJANGO_SUPERUSER_EMAIL not set; skipping superuser creation."
fi

echo "Starting application..."
# Hand off to the CMD from Dockerfile / docker-compose
exec "$@"
