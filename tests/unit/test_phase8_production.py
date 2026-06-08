# ============================================================
# tests/unit/test_phase8_production.py
# Unit tests for Phase 8.3 — Production Pipeline
# Tests: logger, Dockerfile existence, deploy stubs,
#        CI workflow YAML validity
# ============================================================

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


# ── Fixtures ─────────────────────────────────────────────────
@pytest.fixture
def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


# ─────────────────────────────────────────────────────────────
# 1. Logger Tests
# ─────────────────────────────────────────────────────────────
class TestLogger:
    def test_logger_import(self):
        """Logger module should be importable."""
        from src.core.logger import get_logger
        assert callable(get_logger)

    def test_get_logger_returns_logger(self):
        from src.core.logger import get_logger
        log = get_logger("test.phase8")
        assert isinstance(log, logging.Logger)

    def test_get_logger_has_handlers(self):
        from src.core.logger import get_logger
        log = get_logger("test.phase8.handlers")
        assert len(log.handlers) > 0

    def test_logger_does_not_crash_on_info(self, tmp_path, monkeypatch):
        """Logger should not raise when logging info message."""
        monkeypatch.setenv("LOG_DIR", str(tmp_path))
        monkeypatch.setenv("ENV", "development")
        from importlib import reload
        import src.core.logger as lg_module
        reload(lg_module)
        log = lg_module.get_logger("test.no_crash")
        # Should not raise
        log.info("Test info message", extra={"doc_id": "test-001"})

    def test_pipeline_logger_returns_logger(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LOG_DIR", str(tmp_path))
        from importlib import reload
        import src.core.logger as lg_module
        reload(lg_module)
        log = lg_module.get_pipeline_logger("test.pipeline")
        assert isinstance(log, logging.Logger)

    def test_json_formatter_output(self):
        from src.core.logger import JSONFormatter
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test", level=logging.INFO,
            pathname="", lineno=0,
            msg="hello world", args=(), exc_info=None
        )
        result = formatter.format(record)
        parsed = json.loads(result)
        assert parsed["level"] == "INFO"
        assert parsed["message"] == "hello world"
        assert "timestamp" in parsed


# ─────────────────────────────────────────────────────────────
# 2. Docker File Existence Tests
# ─────────────────────────────────────────────────────────────
class TestDockerFiles:
    def test_dockerfile_prod_exists(self, project_root):
        assert (project_root / "docker" / "Dockerfile.prod").exists(), \
            "docker/Dockerfile.prod must exist for Phase 8.3"

    def test_docker_compose_prod_exists(self, project_root):
        assert (project_root / "docker" / "docker-compose.prod.yml").exists(), \
            "docker/docker-compose.prod.yml must exist"

    def test_dockerfile_prod_has_healthcheck(self, project_root):
        content = (project_root / "docker" / "Dockerfile.prod").read_text()
        assert "HEALTHCHECK" in content, "Dockerfile.prod must define a HEALTHCHECK"

    def test_dockerfile_prod_has_nonroot_user(self, project_root):
        content = (project_root / "docker" / "Dockerfile.prod").read_text()
        assert "USER realprop" in content, \
            "Dockerfile.prod must run as non-root user for security"

    def test_dockerfile_prod_multistage(self, project_root):
        content = (project_root / "docker" / "Dockerfile.prod").read_text()
        assert content.count("FROM ") >= 2, \
            "Dockerfile.prod should use multi-stage build"


# ─────────────────────────────────────────────────────────────
# 3. Deploy (Terraform) Stub Tests
# ─────────────────────────────────────────────────────────────
class TestDeployStubs:
    def test_deploy_folder_exists(self, project_root):
        assert (project_root / "deploy").is_dir(), "deploy/ folder must exist"

    def test_terraform_main_exists(self, project_root):
        assert (project_root / "deploy" / "main.tf").exists()

    def test_terraform_variables_exists(self, project_root):
        assert (project_root / "deploy" / "variables.tf").exists()

    def test_terraform_ecs_exists(self, project_root):
        assert (project_root / "deploy" / "ecs.tf").exists()

    def test_terraform_s3_exists(self, project_root):
        assert (project_root / "deploy" / "s3.tf").exists()

    def test_terraform_rds_exists(self, project_root):
        assert (project_root / "deploy" / "rds.tf").exists()

    def test_rds_disabled_for_mvp(self, project_root):
        rds_content = (project_root / "deploy" / "rds.tf").read_text()
        assert "count = 0" in rds_content, \
            "RDS must be disabled (count = 0) for MVP"

    def test_deploy_readme_exists(self, project_root):
        assert (project_root / "deploy" / "README.md").exists()


# ─────────────────────────────────────────────────────────────
# 4. GitHub Actions Workflow Tests
# ─────────────────────────────────────────────────────────────
class TestCIWorkflow:
    def test_ci_workflow_exists(self, project_root):
        ci_path = project_root / ".github" / "workflows" / "ci.yml"
        assert ci_path.exists(), ".github/workflows/ci.yml must exist"

    def test_ci_workflow_is_valid_yaml(self, project_root):
        import yaml
        ci_path = project_root / ".github" / "workflows" / "ci.yml"
        with open(ci_path) as f:
            content = f.read()
            parsed = yaml.safe_load(content)
        assert "jobs" in parsed, "CI workflow must define jobs"
        # PyYAML parses bare `on:` as boolean True — check both forms
        assert ("on" in parsed or True in parsed), \
                 "CI workflow must define triggers (on:)"
        # Also verify the raw text contains the 'on:' trigger block
        assert "on:" in content, "CI workflow must have 'on:' trigger section in raw YAML"

    def test_ci_has_test_job(self, project_root):
        import yaml
        ci_path = project_root / ".github" / "workflows" / "ci.yml"
        with open(ci_path) as f:
            parsed = yaml.safe_load(f)
        assert "test" in parsed["jobs"], "CI workflow must have a 'test' job"

    def test_ci_has_docker_build_job(self, project_root):
        import yaml
        ci_path = project_root / ".github" / "workflows" / "ci.yml"
        with open(ci_path) as f:
            parsed = yaml.safe_load(f)
        assert "docker-build" in parsed["jobs"], \
            "CI workflow must have a 'docker-build' job"

    def test_ci_docker_job_depends_on_test(self, project_root):
        import yaml
        ci_path = project_root / ".github" / "workflows" / "ci.yml"
        with open(ci_path) as f:
            parsed = yaml.safe_load(f)
        needs = parsed["jobs"]["docker-build"].get("needs", [])
        assert "test" in needs, \
            "docker-build job must depend on test job (needs: test)"