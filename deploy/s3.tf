# ============================================================
# deploy/s3.tf  — S3 Buckets for Documents + MLflow Artifacts
# ============================================================

locals {
  account_id = data.aws_caller_identity.current.account_id
  region     = data.aws_region.current.name
}

# ── Document Storage Bucket ──────────────────────────────────
resource "aws_s3_bucket" "documents" {
  bucket = "realprop-documents-${local.account_id}-${var.environment}"

  # TODO: Enable versioning for document audit trail
  # TODO: Add S3 Object Lock for legal compliance (WORM)
}

resource "aws_s3_bucket_versioning" "documents" {
  bucket = aws_s3_bucket.documents.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "documents" {
  bucket = aws_s3_bucket.documents.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_public_access_block" "documents" {
  bucket                  = aws_s3_bucket.documents.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# ── MLflow Artifact Bucket ───────────────────────────────────
resource "aws_s3_bucket" "mlflow_artifacts" {
  bucket = "realprop-mlflow-artifacts-${local.account_id}-${var.environment}"
}

resource "aws_s3_bucket_server_side_encryption_configuration" "mlflow_artifacts" {
  bucket = aws_s3_bucket.mlflow_artifacts.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_public_access_block" "mlflow_artifacts" {
  bucket                  = aws_s3_bucket.mlflow_artifacts.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}