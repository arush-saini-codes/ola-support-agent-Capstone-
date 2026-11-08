"""Prompt Injection Guardrail (Section 12.6).

Detects adversarial prompt injection attempts using regex patterns over
normalized text. Blocked requests are intercepted before reaching agents.
"""

import re

INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(?:all\s+|any\s+)?(?:previous|prior|above)\s+instructions", re.IGNORECASE),
    re.compile(r"disregard\s+(?:your|the)\s+(?:rules|instructions)", re.IGNORECASE),
    re.compile(r"system\s+prompt", re.IGNORECASE),
    re.compile(r"you\s+are\s+now", re.IGNORECASE),
    re.compile(r"act\s+as", re.IGNORECASE),
    re.compile(r"developer\s+mode", re.IGNORECASE),
    re.compile(r"jailbreak", re.IGNORECASE),
    re.compile(r"reveal\s+(?:your|the)\s+(?:prompt|instructions)", re.IGNORECASE),
    re.compile(r"override\s+(?:your|the|all)", re.IGNORECASE),
    re.compile(r"pretend\s+(?:to\s+be|you\s+are)", re.IGNORECASE),
    re.compile(r"forget\s+(?:everything|your\s+instructions)", re.IGNORECASE),
    re.compile(r"\bbypass\b", re.IGNORECASE),
]


def detect_injection(text: str) -> tuple[bool, str | None]:
    """Check if input contains prompt injection patterns.
    
    Returns:
        tuple[bool, str | None]: (blocked, matched_pattern_text)
    """
    if not text:
        return False, None

    # Normalize whitespace
    normalized = " ".join(text.split())

    for pattern in INJECTION_PATTERNS:
        match = pattern.search(normalized)
        if match:
            return True, match.group(0)

    return False, None


# Backward compatibility alias
check_injection = detect_injection
