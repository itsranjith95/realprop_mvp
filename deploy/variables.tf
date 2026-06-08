# ============================================================
# deploy/variables.tf
# ============================================================

variable "aws_region" {
  description = "AWS region to deploy into"
  type        = string
  default     = "ap-south-1"   # Mumbai — closest to Bengaluru
}

variable "environment" {
  description = "Deployment environment (dev / staging / prod)"
  type        = string
  default     = "dev"
}

variable "app_image_tag" {
  description = "Docker image tag for realprop-app (ECR image URI)"
  type        = string
  default     = "latest"
  # TODO: In CI, pass this as -var="app_image_tag=<ECR_URI>:<git_sha>"
}

variable "app_cpu" {
  description = "Fargate task CPU units (256 = 0.25 vCPU)"
  type        = number
  default     = 512
}

variable "app_memory" {
  description = "Fargate task memory in MiB"
  type        = number
  default     = 1024
}

variable "db_instance_class" {
  description = "RDS instance class (future)"
  type        = string
  default     = "db.t3.micro"
}

variable "db_password" {
  description = "RDS master password (use Secrets Manager in prod)"
  type        = string
  sensitive   = true
  default     = "TODO_CHANGE_ME"
  # TODO: Replace with aws_secretsmanager_secret reference
}