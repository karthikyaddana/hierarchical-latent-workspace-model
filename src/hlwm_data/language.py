from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional, Tuple

from .util import sanitize_unicode


URL_PATTERN = re.compile(r"https?://\S+", re.IGNORECASE)
WHITESPACE_PATTERN = re.compile(r"\s+")
CODE_HINT_PATTERN = re.compile(
    r"(?:^|\s)(?:def|class|function|return|import|from|const|let|var|public|private|if|else|for|while)\b"
    r"|[{};]|```",
    re.IGNORECASE,
)

# Greek mathematical symbols are intentionally not blocked. These ranges target
# scripts that reliably indicate non-English prose in this corpus.
BLOCKED_SCRIPT_RANGES: Tuple[Tuple[int, int], ...] = (
    (0x0400, 0x052F),  # Cyrillic
    (0x0530, 0x058F),  # Armenian
    (0x0590, 0x05FF),  # Hebrew
    (0x0600, 0x06FF),  # Arabic
    (0x0700, 0x074F),  # Syriac
    (0x0750, 0x077F),
    (0x0780, 0x07BF),  # Thaana
    (0x07C0, 0x07FF),  # NKo
    (0x0800, 0x085F),  # Samaritan and Mandaic
    (0x08A0, 0x08FF),
    (0x0900, 0x097F),  # Devanagari
    (0x0980, 0x09FF),  # Bengali
    (0x0A00, 0x0A7F),  # Gurmukhi
    (0x0A80, 0x0AFF),  # Gujarati
    (0x0B00, 0x0B7F),  # Odia
    (0x0B80, 0x0BFF),  # Tamil
    (0x0C00, 0x0C7F),  # Telugu
    (0x0C80, 0x0CFF),  # Kannada
    (0x0D00, 0x0D7F),  # Malayalam
    (0x0D80, 0x0DFF),  # Sinhala
    (0x0E00, 0x0E7F),  # Thai
    (0x0E80, 0x0EFF),  # Lao
    (0x0F00, 0x0FFF),  # Tibetan
    (0x1000, 0x109F),  # Myanmar
    (0x10A0, 0x10FF),  # Georgian
    (0x1100, 0x11FF),  # Hangul Jamo
    (0x1200, 0x137F),  # Ethiopic
    (0x13A0, 0x167F),  # Cherokee and Canadian Aboriginal syllabics
    (0x1780, 0x18AF),  # Khmer and Mongolian
    (0x2C80, 0x2CFF),  # Coptic
    (0x2D00, 0x2D7F),  # Georgian supplement and Tifinagh
    (0x3040, 0x30FF),  # Hiragana and Katakana
    (0x3100, 0x312F),  # Bopomofo
    (0x3400, 0x4DBF),  # CJK extension A
    (0x4E00, 0x9FFF),  # CJK unified ideographs
    (0xA000, 0xA4CF),  # Yi
    (0xAC00, 0xD7AF),  # Hangul
    (0xF900, 0xFAFF),  # CJK compatibility ideographs
    (0x20000, 0x2FA1F),  # CJK extensions and compatibility supplement
)


@dataclass(frozen=True)
class LanguageDecision:
    language: str
    confidence: float
    accepted: bool
    reason: str


def _is_blocked_script(character: str) -> bool:
    value = ord(character)
    return any(start <= value <= end for start, end in BLOCKED_SCRIPT_RANGES)


def _script_counts(text: str) -> Tuple[int, int]:
    letters = 0
    blocked = 0
    for character in text:
        if not unicodedata.category(character).startswith("L"):
            continue
        letters += 1
        blocked += int(_is_blocked_script(character))
    return letters, blocked


@lru_cache(maxsize=1)
def _identifier():
    try:
        from langid.langid import LanguageIdentifier, model
    except ImportError as exc:
        raise RuntimeError(
            "English-only filtering requires langid. Install project dependencies with: pip install -e ."
        ) from exc
    return LanguageIdentifier.from_modelstring(model, norm_probs=True)


def _sample(text: str, maximum_chars: int = 12000) -> str:
    value = URL_PATTERN.sub(" ", text)
    value = WHITESPACE_PATTERN.sub(" ", value).strip()
    if len(value) <= maximum_chars:
        return value
    # Preserve the beginning, middle and end so a translated tail cannot hide
    # behind a long English prefix.
    section = maximum_chars // 3
    middle = max(0, (len(value) - section) // 2)
    return " ".join((value[:section], value[middle : middle + section], value[-section:]))


def classify_language(
    text: str,
    expected: str = "en",
    minimum_confidence: float = 0.78,
) -> LanguageDecision:
    sample = _sample(sanitize_unicode(str(text)))
    letters, blocked = _script_counts(sample)
    if letters < 24:
        return LanguageDecision("und", 0.0, False, "too_little_natural_language")
    if blocked:
        return LanguageDecision("non_en_script", 1.0, False, "blocked_script_detected")

    language, confidence = _identifier().classify(sample)
    confidence = float(confidence)
    if language == expected and confidence >= minimum_confidence:
        return LanguageDecision(language, confidence, True, "language_model_match")

    # Source-code-heavy records are language-neutral training material. Accept
    # ASCII code only when no non-English script is present; prose still has to
    # pass the language model above.
    ascii_ratio = sum(ord(character) < 128 for character in sample) / max(1, len(sample))
    punctuation_ratio = sum(not character.isalnum() and not character.isspace() for character in sample) / max(
        1, len(sample)
    )
    code_like = bool(CODE_HINT_PATTERN.search(sample)) and punctuation_ratio >= 0.08
    if expected == "en" and code_like and ascii_ratio >= 0.98:
        return LanguageDecision("en", max(confidence, 0.80), True, "ascii_code_with_no_blocked_script")
    return LanguageDecision(language, confidence, False, "language_model_mismatch")


def contains_blocked_script(text: str) -> bool:
    _, blocked = _script_counts(sanitize_unicode(str(text)))
    return blocked > 0
