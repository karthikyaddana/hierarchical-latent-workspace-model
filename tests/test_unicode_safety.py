import json

from hlwm_data.language import classify_language
from hlwm_data.util import normalize_text, sanitize_json_value, sanitize_unicode, stable_hash


def test_sanitize_unicode_combines_valid_surrogate_pair_and_replaces_lone_surrogate():
    value = "math \ud835\udc9c broken \ud835 end"

    assert sanitize_unicode(value) == "math 𝒜 broken \ufffd end"
    assert normalize_text(value) == "math 𝒜 broken \ufffd end"


def test_surrogate_containing_records_are_hashable_and_utf8_serializable():
    value = {"text": "broken \ud835", "nested": ["valid", "low \udc00"]}
    cleaned = sanitize_json_value(value)

    assert stable_hash(value, 32) == stable_hash(cleaned, 32)
    encoded = json.dumps(cleaned, ensure_ascii=False).encode("utf-8")
    assert "\ufffd".encode("utf-8") in encoded


def test_language_classifier_sanitizes_lone_surrogates_before_langid():
    text = (
        "A verifier checks every proposed algorithm against explicit boundary cases "
        "and reports unsupported conclusions "
        + "\ud835"
    )

    decision = classify_language(text)

    assert decision.language == "en"
    assert decision.accepted
