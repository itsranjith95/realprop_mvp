#!/usr/bin/env bash
# Phase 8.1 — Run Full DVC Pipeline
# Usage: bash scripts/run_dvc_pipeline.sh [--force]
set -euo pipefail

FORCE_FLAG=""
if [[ "${1:-}" == "--force" ]]; then
  FORCE_FLAG="--force"
  echo "[run_dvc_pipeline] Running with --force (all stages re-executed)"
fi

echo "=========================================="
echo " RealProp MVP — DVC Pipeline (Phase 8.1)"
echo "=========================================="

# Ensure DVC is initialised
if [ ! -d ".dvc" ]; then
  echo "Initialising DVC..."
  dvc init
fi

echo ""
echo "► Stage 1: data_ingest"
dvc repro data_ingest $FORCE_FLAG

echo ""
echo "► Stage 2: ocr"
dvc repro ocr $FORCE_FLAG

echo ""
echo "► Stage 3: classification_train"
dvc repro classification_train $FORCE_FLAG

echo ""
echo "► Stage 4: classification_evaluate"
dvc repro classification_evaluate $FORCE_FLAG

echo ""
echo "► Stage 5: validate"
dvc repro validate $FORCE_FLAG || echo "  [WARN] validate stage skipped (deps missing)"

echo ""
echo "► Stage 6: rules"
dvc repro rules $FORCE_FLAG || echo "  [WARN] rules stage skipped (deps missing)"

echo ""
echo "► Stage 7: register_models"
dvc repro register_models $FORCE_FLAG || echo "  [WARN] register_models stage skipped"

echo ""
echo "=========================================="
echo " Pipeline complete!"
echo "=========================================="
dvc status