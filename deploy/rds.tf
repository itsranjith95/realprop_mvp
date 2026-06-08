# ============================================================
# deploy/rds.tf  — RDS PostgreSQL (Future — Disabled for MVP)
# Set count = 1 in post-MVP when database is needed.
# ============================================================

# DB Subnet Group (disabled for MVP)
resource "aws_db_subnet_group" "realprop" {
  count      = 0   # TODO: set to 1 when enabling RDS
  name       = "realprop-${var.environment}"
  subnet_ids = ["TODO_PRIVATE_SUBNET_1", "TODO_PRIVATE_SUBNET_2"]
  description = "RealProp RDS subnet group"
}

# RDS PostgreSQL Instance (disabled for MVP)
resource "aws_db_instance" "realprop" {
  count = 0   # TODO: set to 1 when enabling RDS (post-MVP)

  identifier           = "realprop-${var.environment}"
  engine               = "postgres"
  engine_version       = "15.4"
  instance_class       = var.db_instance_class
  allocated_storage    = 20
  storage_encrypted    = true
  db_name              = "realprop"
  username             = "realprop_admin"
  password             = var.db_password
  db_subnet_group_name = aws_db_subnet_group.realprop[0].name
  skip_final_snapshot  = true   # TODO: set false in production

  # TODO: Add parameter group for pg_stat_statements, log_min_duration_statement
  # TODO: Enable Performance Insights
  # TODO: Enable automated backups (backup_retention_period = 7)
  # TODO: Add CloudWatch alarm on DatabaseConnections metric
}

# Output (disabled for MVP)
output "rds_endpoint" {
  value       = length(aws_db_instance.realprop) > 0 ? aws_db_instance.realprop[0].endpoint : "RDS disabled — enable by setting count = 1 in rds.tf"
  description = "RDS PostgreSQL connection endpoint"
}