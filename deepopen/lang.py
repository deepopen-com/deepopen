"""Dependency-free language/script detection used to route between Laya checkpoints.

Routing only needs one decision: *is this English Latin text, or is it something the English
checkpoint cannot read?* Benchmarks on MASSIVE (14 languages) showed the English checkpoint
collapsing to near-random on non-Latin scripts (Hindi 0.100, Korean 0.103, Swahili 0.103,
Tamil 0.113 at 20 options, where random is 0.050), while holding up far better on Latin-script
languages (French 0.487, Spanish 0.480). So the signal that matters most is *script*, and the
secondary signal is whether Latin text is English.

Script detection is exact. The Latin-script language guess is a stopword/diacritic heuristic and
is explicitly best-effort: pass an explicit model or `lang=` when you already know the language.
"""
import re
from typing import Dict, List, Optional, Union

# Unicode blocks that the English (ModernBERT-large, 50k English BPE) checkpoint cannot read.
_SCRIPT_RANGES = [
    ("greek", ((0x0370, 0x03FF), (0x1F00, 0x1FFF))),
    ("cyrillic", ((0x0400, 0x052F), (0x2DE0, 0x2DFF), (0xA640, 0xA69F))),
    ("armenian", ((0x0530, 0x058F),)),
    ("hebrew", ((0x0590, 0x05FF),)),
    ("arabic", ((0x0600, 0x06FF), (0x0750, 0x077F), (0x08A0, 0x08FF), (0xFB50, 0xFDFF), (0xFE70, 0xFEFF))),
    ("devanagari", ((0x0900, 0x097F), (0xA8E0, 0xA8FF))),
    ("bengali", ((0x0980, 0x09FF),)),
    ("gurmukhi", ((0x0A00, 0x0A7F),)),
    ("gujarati", ((0x0A80, 0x0AFF),)),
    ("oriya", ((0x0B00, 0x0B7F),)),
    ("tamil", ((0x0B80, 0x0BFF),)),
    ("telugu", ((0x0C00, 0x0C7F),)),
    ("kannada", ((0x0C80, 0x0CFF),)),
    ("malayalam", ((0x0D00, 0x0D7F),)),
    ("sinhala", ((0x0D80, 0x0DFF),)),
    ("thai", ((0x0E00, 0x0E7F),)),
    ("lao", ((0x0E80, 0x0EFF),)),
    ("tibetan", ((0x0F00, 0x0FFF),)),
    ("myanmar", ((0x1000, 0x109F),)),
    ("georgian", ((0x10A0, 0x10FF),)),
    ("ethiopic", ((0x1200, 0x137F),)),
    ("khmer", ((0x1780, 0x17FF),)),
    ("hangul", ((0x1100, 0x11FF), (0x3130, 0x318F), (0xAC00, 0xD7AF))),
    ("kana", ((0x3040, 0x309F), (0x30A0, 0x30FF), (0x31F0, 0x31FF))),
    ("han", ((0x3400, 0x4DBF), (0x4E00, 0x9FFF), (0xF900, 0xFAFF))),
]

# Function words. Latin-script languages overlap heavily (de/la/le/un/e/que), so each hit is
# weighted and a margin is required before calling something non-English.
_STOP = {
    "en": {"the", "and", "is", "are", "was", "were", "to", "of", "in", "for", "with", "that",
           "this", "it", "you", "have", "has", "not", "but", "on", "at", "be", "as", "from",
           "will", "can", "would", "there", "their", "what", "which", "please", "we", "i"},
    "fr": {"le", "la", "les", "des", "une", "est", "pour", "dans", "que", "qui", "avec", "sur",
           "pas", "plus", "nous", "vous", "être", "cette", "mais", "sont", "ont", "aux", "ce"},
    "de": {"der", "die", "das", "und", "ist", "ein", "eine", "den", "dem", "nicht", "mit", "für",
           "auf", "von", "zu", "sich", "auch", "werden", "wurde", "haben", "sind", "oder", "aber"},
    "es": {"el", "los", "las", "que", "por", "con", "para", "una", "es", "se", "del", "como",
           "pero", "son", "está", "este", "esta", "todo", "más", "muy", "hay", "sus"},
    "pt": {"os", "as", "que", "em", "um", "uma", "para", "com", "não", "é", "se", "do", "da",
           "dos", "das", "mas", "são", "está", "este", "esta", "muito", "pelo", "pela"},
    "it": {"il", "lo", "gli", "che", "di", "per", "con", "non", "è", "si", "del", "della", "sono",
           "questo", "questa", "anche", "come", "più", "sono", "nella", "alla"},
    "nl": {"het", "een", "van", "is", "op", "te", "dat", "niet", "met", "voor", "zijn", "aan",
           "door", "maar", "ook", "worden", "deze", "naar", "wordt"},
}
_NON_EN_DIACRITICS = set("àâäãáåçéèêëíìîïñóòôöõøúùûüýÿßæœđłşţğıåäö")
_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)


def _iter_text(state: Union[str, dict, list, None], _depth: int = 0) -> List[str]:
    """Collect the string leaves of a state (str / dict / list), so detection sees real content."""
    if _depth > 6 or state is None:
        return []
    if isinstance(state, str):
        return [state]
    if isinstance(state, dict):
        out = []
        for v in state.values():
            out.extend(_iter_text(v, _depth + 1))
        return out
    if isinstance(state, (list, tuple)):
        out = []
        for v in state:
            out.extend(_iter_text(v, _depth + 1))
        return out
    return []


def state_text(state: Union[str, dict, list, None], max_chars: int = 4000) -> str:
    """Flatten a state into the text used for detection (keys are ignored: they are usually English)."""
    return " ".join(_iter_text(state))[:max_chars]


def detect_script(text: str) -> str:
    """Dominant script of `text`: 'latin', 'han', 'devanagari', ... or 'unknown' if there are no letters."""
    counts: Dict[str, int] = {}
    latin = 0
    for ch in text:
        if not ch.isalpha():
            continue
        cp = ord(ch)
        if cp < 0x0250 or 0x1E00 <= cp <= 0x1EFF:      # Latin + Latin Extended Additional
            latin += 1
            continue
        for name, ranges in _SCRIPT_RANGES:
            if any(lo <= cp <= hi for lo, hi in ranges):
                counts[name] = counts.get(name, 0) + 1
                break
    counts["latin"] = latin
    total = sum(counts.values())
    if total == 0:
        return "unknown"
    return max(counts.items(), key=lambda kv: kv[1])[0]


def script_profile(text: str) -> Dict[str, float]:
    """Fraction of alphabetic characters belonging to each detected script."""
    counts: Dict[str, int] = {"latin": 0}
    for ch in text:
        if not ch.isalpha():
            continue
        cp = ord(ch)
        if cp < 0x0250 or 0x1E00 <= cp <= 0x1EFF:
            counts["latin"] += 1
            continue
        for name, ranges in _SCRIPT_RANGES:
            if any(lo <= cp <= hi for lo, hi in ranges):
                counts[name] = counts.get(name, 0) + 1
                break
    total = sum(counts.values())
    if not total:
        return {}
    return {k: v / total for k, v in counts.items() if v}


def guess_latin_language(text: str) -> Optional[str]:
    """Best-effort language code for Latin-script text, or None when undecided.

    Scores function-word hits per language and requires the winner to beat English by a margin,
    so ordinary English is never misrouted. Short inputs usually return None on purpose.
    """
    words = [w.lower() for w in _WORD.findall(text)]
    if len(words) < 4:
        return None
    scores = {lg: sum(1 for w in words if w in sw) for lg, sw in _STOP.items()}
    lowered = text.lower()
    diac = sum(1 for ch in lowered if ch in _NON_EN_DIACRITICS)
    diac_rate = diac / max(1, len(lowered))
    en = scores.get("en", 0)
    best_lg, best = max(((lg, s) for lg, s in scores.items() if lg != "en"),
                        key=lambda kv: kv[1], default=(None, 0))
    if best == 0 and diac_rate < 0.02:
        return "en" if en else None
    # a non-English language needs a clear margin over English function words
    if best_lg and best >= max(2, en + 2):
        return best_lg
    if diac_rate >= 0.04 and best_lg and best >= en:
        return best_lg
    return "en" if en else None


def analyse(state: Union[str, dict, list, None]) -> Dict[str, object]:
    """Full detection result for a state.

    Returns `script`, `script_profile`, `language` (best effort, may be None),
    `is_english` and `non_latin_fraction`.
    """
    text = state_text(state)
    prof = script_profile(text)
    script = detect_script(text)
    non_latin = round(1.0 - prof.get("latin", 0.0), 4) if prof else 0.0
    if script == "unknown":
        return {"script": "unknown", "script_profile": prof, "language": None,
                "is_english": True, "non_latin_fraction": 0.0}
    if script != "latin":
        return {"script": script, "script_profile": prof, "language": None,
                "is_english": False, "non_latin_fraction": non_latin}
    lang = guess_latin_language(text)
    return {"script": "latin", "script_profile": prof, "language": lang,
            "is_english": lang in (None, "en"), "non_latin_fraction": non_latin}


def is_english(state: Union[str, dict, list, None]) -> bool:
    """True when the English checkpoint can be expected to read this state."""
    return bool(analyse(state)["is_english"])
