# ============================================================
# deploy/ecs.tf  — ECS Fargate Cluster + Service
# MVP NOTE: Simplified sketch — VPC/subnets use default AWS VPC.
# TODO: Create dedicated VPC with private subnets for production.
# ============================================================

# ── ECR Repository ───────────────────────────────────────────
resource "aws_ecr_repository" "realprop_app" {
  name                 = "realprop-app"
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true  # TODO: Add Inspector findings alert via EventBridge
  }
}

# ── ECS Cluster ──────────────────────────────────────────────
resource "aws_ecs_cluster" "realprop" {
  name = "realprop-${var.environment}"

  setting {
    name  = "containerInsights"
    value = "enabled"
    # TODO: CloudWatch Container Insights will send metrics here
  }
}

# ── IAM Role for ECS Task ────────────────────────────────────
resource "aws_iam_role" "ecs_task_execution" {
  name = "realprop-ecs-execution-${var.environment}"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action    = "sts:AssumeRole"
      Effect    = "Allow"
      Principal = { Service = "ecs-tasks.amazonaws.com" }
    }]
  })
}

resource "aws_iam_role_policy_attachment" "ecs_execution_policy" {
  role       = aws_iam_role.ecs_task_execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

# S3 access for the app task
resource "aws_iam_role_policy" "ecs_s3_access" {
  name = "realprop-s3-access"
  role = aws_iam_role.ecs_task_execution.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect   = "Allow"
      Action   = ["s3:GetObject", "s3:PutObject", "s3:DeleteObject", "s3:ListBucket"]
      Resource = [
        aws_s3_bucket.documents.arn,
        "${aws_s3_bucket.documents.arn}/*",
        aws_s3_bucket.mlflow_artifacts.arn,
        "${aws_s3_bucket.mlflow_artifacts.arn}/*"
      ]
    }]
  })
}

# ── CloudWatch Log Group ─────────────────────────────────────
resource "aws_cloudwatch_log_group" "realprop_app" {
  name              = "/ecs/realprop-app-${var.environment}"
  retention_in_days = 14
  # TODO: Add metric filter for ERROR log pattern → SNS alert
}

# ── ECS Task Definition ──────────────────────────────────────
resource "aws_ecs_task_definition" "realprop_app" {
  family                   = "realprop-app-${var.environment}"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = var.app_cpu
  memory                   = var.app_memory
  execution_role_arn       = aws_iam_role.ecs_task_execution.arn
  task_role_arn            = aws_iam_role.ecs_task_execution.arn

  container_definitions = jsonencode([
    {
      name  = "realprop-app"
      image = "${aws_ecr_repository.realprop_app.repository_url}:${var.app_image_tag}"
      portMappings = [
        { containerPort = 8501, protocol = "tcp" },  # Streamlit
        { containerPort = 8000, protocol = "tcp" }   # FastAPI
      ]
      environment = [
        { name = "ENV",                  value = var.environment },
        { name = "MLFLOW_TRACKING_URI",  value = "sqlite:///mlflow.db" },
        # TODO: Replace with RDS URI when db module is enabled
        # TODO: Add OPENAI_API_KEY from AWS Secrets Manager:
        # { name = "OPENAI_API_KEY", valueFrom = "arn:aws:secretsmanager:..." }
      ]
      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = aws_cloudwatch_log_group.realprop_app.name
          "awslogs-region"        = var.aws_region
          "awslogs-stream-prefix" = "app"
        }
        # TODO: Add OpenTelemetry collector sidecar for distributed tracing
      }
      healthCheck = {
        command     = ["CMD-SHELL", "curl -f http://localhost:8000/health || exit 1"]
        interval    = 30
        timeout     = 5
        retries     = 3
        startPeriod = 60
      }
    }
  ])
}

# ── ECS Service ──────────────────────────────────────────────
# TODO: Replace subnet_ids and security_group_ids with actual values
# or use data sources to look up the default VPC.
resource "aws_ecs_service" "realprop_app" {
  name            = "realprop-app-${var.environment}"
  cluster         = aws_ecs_cluster.realprop.id
  task_definition = aws_ecs_task_definition.realprop_app.arn
  desired_count   = 1
  launch_type     = "FARGATE"

  network_configuration {
    subnets          = ["TODO_SUBNET_ID_1", "TODO_SUBNET_ID_2"]
    security_groups  = ["TODO_SECURITY_GROUP_ID"]
    assign_public_ip = true   # TODO: set false + add NAT gateway in prod
  }

  # TODO: Add load_balancer block once ALB is created
  # load_balancer {
  #   target_group_arn = aws_lb_target_group.realprop.arn
  #   container_name   = "realprop-app"
  #   container_port   = 8501
  # }

  lifecycle {
    ignore_changes = [task_definition]
    # Allows CI/CD to update task definition without Terraform drift
  }
}