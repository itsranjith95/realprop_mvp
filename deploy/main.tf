# ============================================================
# deploy/main.tf  — RealProp MVP AWS Infrastructure Sketch
# MVP NOTE: This is a documented stub. Not yet wired to real AWS.
# TODO: Replace all TODO_ placeholders before applying.
# ============================================================

terraform {
  required_version = ">= 1.6.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }

  # TODO (post-MVP): Enable S3 backend for shared state
  # backend "s3" {
  #   bucket = "realprop-terraform-state"
  #   key    = "mvp/terraform.tfstate"
  #   region = var.aws_region
  # }
}

provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Project     = "RealProp"
      Environment = var.environment
      ManagedBy   = "Terraform"
    }
  }
}

# ── Data Sources ─────────────────────────────────────────────
data "aws_caller_identity" "current" {}
data "aws_region" "current" {}