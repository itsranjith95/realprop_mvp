# test_phase5_smoke.py  ← place this at realprop_mvp/
from logs.logging_config import setup_logging
setup_logging()   # activates all log files in logs/ folder

from src.pipelines.validation_pipeline import collect_validation_results
from src.pipelines.rules_pipeline import run_rules_pipeline

# --- Scenario 1: Perfect match ---
good_case = {
    "mother_deed": {
        "buyer_name": "Ranjith Kumar",
        "seller_name": "Suresh Reddy",
        "property_id": "SY NO 45/2",
        "area": "1200 sqft",
    },
    "khata": {
        "owner_name": "Ranjith Kumar",
        "property_id": "SY NO 45/2",
        "area": "1200 sqft",
    },
}

validations = collect_validation_results("CASE_GOOD_001", good_case)
output = run_rules_pipeline("CASE_GOOD_001", good_case, validations, persist=True)

print("\n--- GOOD CASE ---")
print(f"Score      : {output.risk_score}")
print(f"Label      : {output.risk_label.value}")
print(f"Must Review: {output.mandatory_review}")
print(f"Summary    : {output.summary}")


# --- Scenario 2: Mismatches ---
bad_case = {
    "mother_deed": {
        "buyer_name": "Ranjith Kumar",
        "seller_name": "Suresh Reddy",
        "property_id": "SY NO 45/2",
        "area": "1200 sqft",
    },
    "khata": {
        "owner_name": "Mahesh Babu",        # mismatch
        "property_id": "SY NO 99/1",        # mismatch
        "area": "900 sqft",                 # mismatch
    },
}

validations_bad = collect_validation_results("CASE_BAD_001", bad_case)
output_bad = run_rules_pipeline("CASE_BAD_001", bad_case, validations_bad, persist=True)

print("\n--- BAD CASE ---")
print(f"Score      : {output_bad.risk_score}")
print(f"Label      : {output_bad.risk_label.value}")
print(f"Must Review: {output_bad.mandatory_review}")
print(f"Summary    : {output_bad.summary}")
print("Rules fired:")
for hit in output_bad.rule_hits:
    print(f"  → {hit.rule_id} | {hit.severity.value} | +{hit.points} pts")


# --- Scenario 3: Missing fields ---
missing_case = {
    "mother_deed": {"buyer_name": None, "seller_name": None, "property_id": None},
    "khata": {"owner_name": None, "property_id": None},
}

validations_missing = collect_validation_results("CASE_MISS_001", missing_case)
output_missing = run_rules_pipeline("CASE_MISS_001", missing_case, validations_missing, persist=True)

print("\n--- MISSING FIELDS CASE ---")
print(f"Score      : {output_missing.risk_score}")
print(f"Label      : {output_missing.risk_label.value}")
print(f"Must Review: {output_missing.mandatory_review}")