# debug_regex.py  — delete after debugging
import re

text = "Khata No. 456/23\nOwner's Name: Surekha Vijay Nair\nProperty ID: 789012"

pat = re.compile(
    r"(?:owner['\u2019\u2018]?s?\s+name|name\s+of\s+(?:the\s+)?owner|owner)\s*[:\-]\s*([A-Za-z][A-Za-z\s.]{3,60}?)(?=\s*(?:,|\r?\n|son\b|d/o|w/o|khata|ward|zone|$))",
    re.IGNORECASE
)
m = pat.search(text)
print("MATCH:", m.group(1) if m else "NO MATCH")