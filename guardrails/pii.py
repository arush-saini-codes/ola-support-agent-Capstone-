"""PII Masking Guardrail for Indian phone numbers (Section 12.6).

Detects and masks Indian mobile numbers with '[PHONE]' prior to
downstream processing (model, memory, cache, and structured logs).
Preserves ticket IDs (OLA-XXXX), currency amounts, and durations.
"""

import re

# Indian mobile phone numbers: optional +91/91/0 prefix, begins with 6-9, followed by 9 digits
PHONE_RE = re.compile(r"(?<![\w])(?:\+?91[\s-]?|0)?[6-9]\d{4}[\s-]?\d{5}(?!\d)")


def mask_pii(text: str) -> tuple[str, int]:
    """Mask Indian phone numbers in input text with '[PHONE]'.
    
    Returns:
        tuple[str, int]: (masked_text, count_of_masked_numbers)
    """
    if not text:
        return text, 0

    masked_text, count = PHONE_RE.subn("[PHONE]", text)
    return masked_text, count

