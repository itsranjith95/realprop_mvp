"""
Unit Tests — Rules Configuration & Rules Service
Tests that:
1. rules_config.yaml is valid and has required fields.
2. Each rule has a rule_id, rule_version, severity, points, condition.
3. The risk scoring thresholds are consistent.
4. Rules service evaluate_rules() triggers correctly on mock data.
"""
import pytest
import yaml
from pathlib import Path

RULES_CONFIG_PATH = Path("config/rules_config.yaml")


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def rules_config():
    assert RULES_CONFIG_PATH.exists(), f"rules_config.yaml not found at {RULES_CONFIG_PATH}"
    with open(RULES_CONFIG_PATH) as f:
        return yaml.safe_load(f)


# ─── Meta / schema tests ──────────────────────────────────────────────────────

class TestRulesConfigSchema:

    def test_has_meta_section(self, rules_config):
        assert "meta" in rules_config, "Missing 'meta' section"

    def test_meta_has_version(self, rules_config):
        meta = rules_config["meta"]
        assert "rule_set_version" in meta, "meta must have rule_set_version"
        assert meta["rule_set_version"], "rule_set_version must not be empty"

    def test_has_rules_list(self, rules_config):
        assert "rules" in rules_config, "Missing 'rules' list"
        assert isinstance(rules_config["rules"], list), "'rules' must be a list"
        assert len(rules_config["rules"]) > 0, "'rules' list must not be empty"

    def test_has_risk_scoring(self, rules_config):
        assert "risk_scoring" in rules_config, "Missing 'risk_scoring' section"
        rs = rules_config["risk_scoring"]
        for key in ("green_max", "yellow_max", "red_above"):
            assert key in rs, f"risk_scoring missing '{key}'"

    def test_risk_thresholds_consistent(self, rules_config):
        rs = rules_config["risk_scoring"]
        assert rs["green_max"] < rs["yellow_max"], "green_max must be < yellow_max"
        assert rs["yellow_max"] == rs["red_above"], "yellow_max should equal red_above"


# ─── Per-rule field tests ─────────────────────────────────────────────────────

class TestIndividualRules:

    REQUIRED_FIELDS = ["rule_id", "rule_version", "name", "severity", "description",
                       "applies_to", "condition", "points"]
    VALID_SEVERITIES = {"high", "medium", "low"}
    VALID_APPLIES_TO = {"MOTHER_DEED", "KHATA", "CROSS"}

    def test_all_required_fields_present(self, rules_config):
        for rule in rules_config["rules"]:
            for field in self.REQUIRED_FIELDS:
                assert field in rule, f"Rule {rule.get('rule_id','?')} missing '{field}'"

    def test_rule_ids_are_unique(self, rules_config):
        ids = [r["rule_id"] for r in rules_config["rules"]]
        assert len(ids) == len(set(ids)), "Duplicate rule_id found!"

    def test_severity_values_valid(self, rules_config):
        for rule in rules_config["rules"]:
            assert rule["severity"] in self.VALID_SEVERITIES, \
                f"Rule {rule['rule_id']} has invalid severity: {rule['severity']}"

    def test_applies_to_values_valid(self, rules_config):
        for rule in rules_config["rules"]:
            for doc_type in rule["applies_to"]:
                assert doc_type in self.VALID_APPLIES_TO, \
                    f"Rule {rule['rule_id']} has invalid applies_to value: {doc_type}"

    def test_points_are_positive_integers(self, rules_config):
        for rule in rules_config["rules"]:
            pts = rule["points"]
            assert isinstance(pts, int) and pts > 0, \
                f"Rule {rule['rule_id']} points must be a positive int, got: {pts}"

    def test_high_severity_rules_have_requires_review(self, rules_config):
        """All high-severity rules must have requires_review: true."""
        for rule in rules_config["rules"]:
            if rule["severity"] == "high":
                assert rule.get("requires_review") is True, \
                    f"High-severity rule {rule['rule_id']} must have requires_review: true"

    def test_cross_rules_have_fields_list(self, rules_config):
        """Cross-document rules must specify 'fields' (plural) not 'field'."""
        for rule in rules_config["rules"]:
            if "CROSS" in rule.get("applies_to", []):
                assert "fields" in rule, \
                    f"Cross rule {rule['rule_id']} must have 'fields' list"

    def test_fuzzy_rules_have_threshold(self, rules_config):
        """Rules with fuzzy_mismatch condition must define fuzzy_threshold."""
        for rule in rules_config["rules"]:
            if rule.get("condition") == "fuzzy_mismatch":
                assert "fuzzy_threshold" in rule, \
                    f"Rule {rule['rule_id']} (fuzzy_mismatch) missing 'fuzzy_threshold'"
                assert 0 < rule["fuzzy_threshold"] <= 100, \
                    f"Rule {rule['rule_id']} fuzzy_threshold must be in (0, 100]"

    def test_confidence_rules_have_threshold(self, rules_config):
        """Rules that check confidence must define confidence_threshold."""
        for rule in rules_config["rules"]:
            if "confidence" in rule.get("condition", ""):
                assert "confidence_threshold" in rule, \
                    f"Rule {rule['rule_id']} missing 'confidence_threshold'"
                val = rule["confidence_threshold"]
                assert 0.0 < val < 1.0, \
                    f"Rule {rule['rule_id']} confidence_threshold must be in (0, 1)"


# ─── Functional / evaluate_rules tests ───────────────────────────────────────

class TestRulesServiceFunctionality:
    """
    These tests import the rules service and test evaluate_rules logic
    against mock entity / validation data.
    """

    @pytest.fixture(scope="class")
    def rules_service(self):
        """Import rules service if available."""
        try:
            from src.services.rules_service import evaluate_rules, load_rules_config
            return evaluate_rules, load_rules_config
        except ImportError:
            pytest.skip("src.services.rules_service not importable — skipping functional tests")

    def test_load_rules_config_returns_dict(self, rules_service):
        _, load_rules_config = rules_service
        config = load_rules_config()
        assert isinstance(config, dict)
        assert "rules" in config

    def test_no_flags_on_perfect_case(self, rules_service):
        """When all fields match perfectly, no rules should trigger."""
        evaluate_rules, _ = rules_service
        entities = {
            "mother_deed": {
                "buyer_name": {"value": "Ranjith Kumar", "confidence": 0.95},
                "seller_name": {"value": "Shyam Rao", "confidence": 0.95},
                "property_id": {"value": "SY-123/4A", "confidence": 0.95},
                "registration_date": {"value": "2023-08-15", "confidence": 0.95},
            },
            "khata": {
                "owner_name": {"value": "Ranjith Kumar", "confidence": 0.95},
                "khata_number": {"value": "KH-789", "confidence": 0.95},
                "property_id": {"value": "SY-123/4A", "confidence": 0.95},
            },
        }
        result = evaluate_rules("CASE_TEST_001", entities)
        assert result["risk_label"] in ("Low Risk", "Green", "green"), \
            f"Expected green risk for clean case, got: {result['risk_label']}"
        assert result["risk_score"] <= 15

    def test_owner_name_mismatch_triggers_r010(self, rules_service):
        """Mismatched buyer/owner name should trigger R010."""
        evaluate_rules, _ = rules_service
        entities = {
            "mother_deed": {
                "buyer_name": {"value": "Ranjith Kumar", "confidence": 0.95},
                "seller_name": {"value": "Shyam Rao", "confidence": 0.95},
                "property_id": {"value": "SY-123/4A", "confidence": 0.95},
                "registration_date": {"value": "2023-08-15", "confidence": 0.95},
            },
            "khata": {
                "owner_name": {"value": "Rajesh Sharma", "confidence": 0.95},  # DIFFERENT
                "khata_number": {"value": "KH-789", "confidence": 0.95},
                "property_id": {"value": "SY-123/4A", "confidence": 0.95},
            },
        }
        result = evaluate_rules("CASE_TEST_002", entities)
        rule_ids = [h["rule_id"] for h in result.get("rule_hits", [])]
        assert "R010" in rule_ids, "R010 (OWNER_NAME_MISMATCH) should have been triggered"

    def test_missing_owner_name_triggers_r001(self, rules_service):
        """Missing Khata owner name should trigger R001."""
        evaluate_rules, _ = rules_service
        entities = {
            "mother_deed": {
                "buyer_name": {"value": "Ranjith Kumar", "confidence": 0.95},
                "seller_name": {"value": "Shyam Rao", "confidence": 0.95},
                "property_id": {"value": "SY-123/4A", "confidence": 0.95},
                "registration_date": {"value": "2023-08-15", "confidence": 0.95},
            },
            "khata": {
                "owner_name": {"value": "", "confidence": 0.0},  # MISSING
                "khata_number": {"value": "KH-789", "confidence": 0.95},
                "property_id": {"value": "SY-123/4A", "confidence": 0.95},
            },
        }
        result = evaluate_rules("CASE_TEST_003", entities)
        rule_ids = [h["rule_id"] for h in result.get("rule_hits", [])]
        assert "R001" in rule_ids, "R001 (MISSING_OWNER_NAME) should have been triggered"

    def test_property_id_mismatch_triggers_r011(self, rules_service):
        """Property ID mismatch should trigger R011."""
        evaluate_rules, _ = rules_service
        entities = {
            "mother_deed": {
                "buyer_name": {"value": "Ranjith Kumar", "confidence": 0.95},
                "seller_name": {"value": "Shyam Rao", "confidence": 0.95},
                "property_id": {"value": "SY-123/4A", "confidence": 0.95},
                "registration_date": {"value": "2023-08-15", "confidence": 0.95},
            },
            "khata": {
                "owner_name": {"value": "Ranjith Kumar", "confidence": 0.95},
                "khata_number": {"value": "KH-789", "confidence": 0.95},
                "property_id": {"value": "SY-999/1B", "confidence": 0.95},  # MISMATCH
            },
        }
        result = evaluate_rules("CASE_TEST_004", entities)
        rule_ids = [h["rule_id"] for h in result.get("rule_hits", [])]
        assert "R011" in rule_ids, "R011 (PROPERTY_ID_MISMATCH) should have been triggered"

    def test_high_risk_score_gives_red_label(self, rules_service):
        """Multiple mismatches should result in a Red / High Risk label."""
        evaluate_rules, _ = rules_service
        entities = {
            "mother_deed": {
                "buyer_name": {"value": "", "confidence": 0.0},  # Missing
                "seller_name": {"value": "", "confidence": 0.0},  # Missing
                "property_id": {"value": "SY-123/4A", "confidence": 0.95},
                "registration_date": {"value": "", "confidence": 0.0},  # Missing
            },
            "khata": {
                "owner_name": {"value": "", "confidence": 0.0},  # Missing
                "khata_number": {"value": "", "confidence": 0.0},  # Missing
                "property_id": {"value": "SY-999/1B", "confidence": 0.95},  # Mismatch
            },
        }
        result = evaluate_rules("CASE_TEST_005", entities)
        assert result["risk_score"] > 40, "Score should exceed red threshold"
        label = result["risk_label"].lower()
        assert "high" in label or "red" in label, \
            f"Expected high/red risk label, got: {result['risk_label']}"