#!/usr/bin/env bash
# ============================================================================
# Run pending PostgreSQL migrations against the running nexus-postgres container.
# Checks schema_version table and only applies migrations not yet applied.
# ============================================================================
set -euo pipefail

CONTAINER="${POSTGRES_CONTAINER:-nexus-postgres}"
DB_USER="${POSTGRES_USER:-nexus}"
DB_NAME="${POSTGRES_DB:-nexus}"
INIT_DIR="/docker-entrypoint-initdb.d"

echo "Running migrations on $CONTAINER..."

# Wait for postgres to be ready (max 30s)
for i in $(seq 1 30); do
  if docker exec "$CONTAINER" sh -c "pg_isready -U $DB_USER -d $DB_NAME" >/dev/null 2>&1; then
    break
  fi
  if [ "$i" -eq 30 ]; then
    echo "ERROR: PostgreSQL not ready after 30s"
    exit 1
  fi
  echo "  Waiting for PostgreSQL... ($i/30)"
  sleep 1
done

# Get current schema version (0 if table doesn't exist yet = fresh DB)
CURRENT_VERSION=$(docker exec "$CONTAINER" sh -c \
  "psql -U $DB_USER -d $DB_NAME -tAc \"SELECT COALESCE(MAX(version),0) FROM schema_version;\"" 2>/dev/null || echo "0")
CURRENT_VERSION=$(echo "$CURRENT_VERSION" | tr -d '[:space:]')
echo "  Current schema version: $CURRENT_VERSION"

APPLIED=0

# Run each SQL file in order, skip already-applied ones
for sql_file in $(docker exec "$CONTAINER" sh -c "ls $INIT_DIR/*.sql 2>/dev/null | sort"); do
  filename=$(basename "$sql_file")
  # Extract version number from filename (e.g., 01-schema.sql -> 1)
  file_version=$(echo "$filename" | grep -oE '^[0-9]+' | sed 's/^0*//')

  if [ -z "$file_version" ]; then
    echo "  Skipping $filename (no version prefix)"
    continue
  fi

  if [ "$file_version" -le "$CURRENT_VERSION" ]; then
    echo "  Skipping $filename (v$file_version already applied)"
    continue
  fi

  echo "  Applying $filename (v$file_version)..."
  docker exec "$CONTAINER" sh -c "psql -U $DB_USER -d $DB_NAME -f $sql_file" 2>&1
  APPLIED=$((APPLIED + 1))
done

if [ "$APPLIED" -eq 0 ]; then
  echo "  No new migrations to apply."
else
  echo "  Applied $APPLIED migration(s)."
fi

# Show final version
FINAL_VERSION=$(docker exec "$CONTAINER" sh -c \
  "psql -U $DB_USER -d $DB_NAME -tAc \"SELECT COALESCE(MAX(version),0) FROM schema_version;\"" 2>/dev/null || echo "?")
echo "Migrations complete. Schema version: $(echo "$FINAL_VERSION" | tr -d '[:space:]')"
