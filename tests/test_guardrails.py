"""
Author: Dhiren Kumar Singh
Unit tests for Chatter Guardrails AI functionality.
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chat import (
    redact_pii,
    parse_llm_output,
    DENIED_PATTERNS,
    PII_PATTERNS,
)


class TestPIIRedaction:
    """Test cases for PII detection and redaction."""

    def test_redact_ssn(self):
        """Test SSN redaction."""
        text = "My SSN is 123-45-6789 and I'm John."
        result, detected = redact_pii(text)
        assert "[REDACTED-SSN]" in result
        assert "SSN" in detected
        assert "123-45-6789" not in result

    def test_redact_credit_card(self):
        """Test credit card number redaction."""
        text = "Card: 4111-1111-1111-1111"
        result, detected = redact_pii(text)
        assert "[REDACTED-CC]" in result
        assert "CREDIT_CARD" in detected
        assert "4111-1111-1111-1111" not in result

    def test_redact_email(self):
        """Test email address redaction."""
        text = "Email me at john.doe@example.com"
        result, detected = redact_pii(text)
        assert "[REDACTED-EMAIL]" in result
        assert "EMAIL" in detected
        assert "john.doe@example.com" not in result

    def test_redact_phone(self):
        """Test phone number redaction."""
        text = "Call me at (555) 123-4567"
        result, detected = redact_pii(text)
        assert "[REDACTED-PHONE]" in result
        assert "PHONE" in detected
        assert "555-123-4567" not in result and "5551234567" not in result

    def test_redact_ip_address(self):
        """Test IP address redaction."""
        text = "Server IP: 192.168.1.100"
        result, detected = redact_pii(text)
        assert "[REDACTED-IP]" in result
        assert "IP_ADDRESS" in detected
        assert "192.168.1.100" not in result

    def test_redact_multiple_pii_types(self):
        """Test detection of multiple PII types in single text."""
        text = "Contact john@test.com at 555-123-4567. SSN: 111-22-3333"
        result, detected = redact_pii(text)
        assert "EMAIL" in detected
        assert "PHONE" in detected
        assert "SSN" in detected
        assert len(detected) == 3

    def test_no_pii_in_clean_text(self):
        """Test that clean text passes through unchanged."""
        text = "This is a normal response with no sensitive data."
        result, detected = redact_pii(text)
        assert result == text
        assert len(detected) == 0

    def test_redact_dob(self):
        """Test date of birth redaction."""
        text = "DOB: 01/15/1990"
        result, detected = redact_pii(text)
        assert "[REDACTED-DOB]" in result
        assert "DOB" in detected


class TestParseLLMOutput:
    """Test cases for LLM output parsing."""

    def test_remove_code_block_markers(self):
        """Test removal of markdown code blocks."""
        text = '```python\nprint("hello")\n```'
        result = parse_llm_output(text)
        assert "```" not in result
        assert 'print("hello")' in result

    def test_remove_inline_code(self):
        """Test removal of inline code markers."""
        text = "Use `pip install` to install"
        result = parse_llm_output(text)
        assert "`" not in result

    def test_strips_whitespace(self):
        """Test whitespace stripping."""
        text = "   \n\n  Hello World  \n  "
        result = parse_llm_output(text)
        assert result.startswith("Hello")
        assert result.endswith("Hello World")

    def test_parse_integration_with_redact(self):
        """Test that parse_llm_output integrates PII redaction."""
        text = "My email is john@example.com"
        result = parse_llm_output(text)
        assert "[REDACTED-EMAIL]" in result
        assert "john@example.com" not in result


class TestDeniedPatterns:
    """Test cases for denied patterns detection."""

    def test_password_detection(self):
        """Test password pattern detection."""
        import re

        patterns = [
            "password: secret123",
            "api_key = abc123",
            "secret: mypassword",
        ]
        for text in patterns:
            matched = any(re.search(p, text, re.IGNORECASE) for p in DENIED_PATTERNS)
            assert matched, f"Should detect pattern in: {text}"

    def test_ssn_in_denied_patterns(self):
        """Test SSN detection in denied patterns."""
        import re

        text = "SSN: 123-45-6789"
        matched = any(re.search(p, text, re.IGNORECASE) for p in DENIED_PATTERNS)
        assert matched

    def test_credit_card_in_denied_patterns(self):
        """Test credit card detection in denied patterns."""
        import re

        text = "Card number 1234567890123456"
        matched = any(re.search(p, text, re.IGNORECASE) for p in DENIED_PATTERNS)
        assert matched


class TestPIIPatternsCompleteness:
    """Test that all expected PII patterns are defined."""

    def test_required_pii_types_present(self):
        """Verify all required PII types are defined."""
        required_types = [
            "SSN",
            "CREDIT_CARD",
            "EMAIL",
            "PHONE",
            "IP_ADDRESS",
        ]
        for pii_type in required_types:
            assert pii_type in PII_PATTERNS, f"Missing PII type: {pii_type}"
            assert len(PII_PATTERNS[pii_type]) == 2

    def test_pii_patterns_have_placeholders(self):
        """Verify all PII patterns have placeholders."""
        for pii_type, (pattern, placeholder) in PII_PATTERNS.items():
            assert placeholder.startswith("[REDACTED-"), (
                f"{pii_type} missing proper placeholder"
            )


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_string(self):
        """Test handling of empty string."""
        result, detected = redact_pii("")
        assert result == ""
        assert len(detected) == 0

    def test_very_long_text(self):
        """Test handling of very long text."""
        long_text = "test@example.com " * 1000
        result, detected = redact_pii(long_text)
        assert "[REDACTED-EMAIL]" in result
        assert len(detected) == 1

    def test_special_characters_in_email(self):
        """Test emails with special characters."""
        emails = [
            "user+tag@domain.com",
            "first.last@company.co.uk",
            "user123@sub.domain.com",
        ]
        for email in emails:
            result, detected = redact_pii(f"Send to {email}")
            assert "[REDACTED-EMAIL]" in result
            assert email not in result

    def test_international_phone_numbers(self):
        """Test international phone number formats."""
        phones = [
            "+1-555-123-4567",
            "001-555-123-4567",
            "+44 20 7946 0958",
        ]
        for phone in phones:
            result, detected = redact_pii(f"Call: {phone}")
            assert "[REDACTED-PHONE]" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
