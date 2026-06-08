# ============================================================
# deploy/outputs.tf
# ============================================================

output "ecr_repository_url" {
  description = "ECR repository URL for pushing Docker images"
  value       = aws_ecr_repository.realprop_app.repository_url
}

output "ecs_cluster_name" {
  description = "ECS cluster name"
  value       = aws_ecs_cluster.realprop.name
}

output "documents_bucket_name" {
  description = "S3 bucket name for property documents"
  value       = aws_s3_bucket.documents.bucket
}

output "mlflow_artifacts_bucket_name" {
  description = "S3 bucket name for MLflow artifacts"
  value       = aws_s3_bucket.mlflow_artifacts.bucket
}

output "cloudwatch_log_group" {
  description = "CloudWatch log group for ECS app container"
  value       = aws_cloudwatch_log_group.realprop_app.name
}