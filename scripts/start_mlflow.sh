#!/usr/bin/env bash
# Phase 8.1 — Start MLflow Tracking Server (local)
# Usage: bash scripts/start_mlflow.sh
set -euo pipefail

MLFLOW_PORT="${MLFLOW_PORT:-5000}"
MLFLOW_BACKEND_STORE="${MLFLOW_BACKEND_STORE:-sqlite:///mlflow.db}"
MLFLOW_ARTIFACT_ROOT="${MLFLOW_ARTIFACT_ROOT:-./mlruns}"

echo "Starting MLflow Tracking Server..."
echo "  Tracking URI  : http://localhost:${MLFLOW_PORT}"
echo "  Backend store : ${MLFLOW_BACKEND_STORE}"
echo "  Artifact root : ${MLFLOW_ARTIFACT_ROOT}"
echo ""

mlflow server \
  --host 0.0.0.0 \
  --port "${MLFLOW_PORT}" \
  --backend-store-uri "${MLFLOW_BACKEND_STORE}" \
  --default-artifact-root "${MLFLOW_ARTIFACT_ROOT}"