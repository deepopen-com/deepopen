"""Email utilities for cleaning and structuring email inputs in laya."""
import re
from typing import Dict, Optional

_QUOTE_HEADERS = [
    re.compile(r"^\s*On .{0,300}wrote:\s*$", re.I),
    re.compile(r"^\s*-{2,}\s*(Original|Forwarded) Message\s*-{2,}", re.I),
    re.compile(r"^\s*_{8,}\s*$"),
    re.compile(r"^\s*From:\s.+$", re.I),
]
_SIGNATURE_MARKERS = [
    re.compile(r"^\s*--\s*$"),
    re.compile(r"^\s*(best|kind|warm|many thanks|thanks|thank you|regards|cheers|sincerely)[\w ,!.]*$", re.I),
    re.compile(r"^\s*sent from my (iphone|android|mobile|ipad)", re.I),
]
_DISCLAIMER = re.compile(
    r"(confidential|intended (solely )?for the (use of the )?(named )?(addressee|recipient)|"
    r"if you (have )?received this (e-?mail|message) in error)",
    re.I,
)


def clean_email_body(body: str, max_chars: int = 3000) -> str:
    """Remove quoted email history, signatures and disclaimers to keep input focused."""
    text = (body or "").replace("\r\n", "\n").replace("\r", "\n").replace("\\n", "\n")
    lines = []
    for line in text.split("\n"):
        if any(p.match(line) for p in _QUOTE_HEADERS) and lines:
            break
        if line.lstrip().startswith(">"):
            continue
        lines.append(line.rstrip())
    cut = len(lines)
    for i in range(max(1, min(int(len(lines) * 0.6), len(lines) - 8)), len(lines)):
        if len(lines[i].strip()) <= 40 and any(p.match(lines[i]) for p in _SIGNATURE_MARKERS):
            cut = i
            break
    lines = lines[:cut]
    paragraphs = [p for p in re.split(r"\n\s*\n", "\n".join(lines)) if not _DISCLAIMER.search(p)]
    text = re.sub(r"[ \t]+", " ", "\n\n".join(p.strip() for p in paragraphs if p.strip()))
    return text[:max_chars]


def email_state(subject: str, body: str, sender: Optional[str] = None, clean: bool = True, **extra) -> Dict:
    """Construct a clean state dictionary for email classification."""
    state = {
        "subject": (subject or "").strip(),
        "body": clean_email_body(body) if clean else (body or ""),
    }
    if sender:
        state["from"] = sender
    state.update({k: v for k, v in extra.items() if v is not None})
    return state


def email_questions(categories: Optional[Dict[str, str]] = None) -> Dict:
    """Standard pre-built questions for email triage."""
    categories = categories or {
        "billing": "invoices, payments, refunds",
        "technical": "bugs, outages, integrations",
        "sales": "pricing, demos, new purchases",
        "security": "phishing, scams, account compromise",
        "hr": "hiring, leave, payroll",
        "other": "none of the above",
    }
    return {
        "category": {
            "type": "choice",
            "instructions": "Which team should handle the email in `body`?",
            "criteria": categories,
        },
        "is_spam": {
            "type": "noul",
            "instructions": "Is this email unsolicited spam or bulk marketing?",
        },
        "is_phishing": {
            "type": "noul",
            "instructions": "Is this email a phishing or scam attempt to steal money, credentials, or personal data?",
            "criteria": {"true": "phishing, scam, or fraud", "false": "a legitimate email"},
        },
        "urgency": {
            "type": "score",
            "instructions": "How urgent is the request in `body`?",
            "criteria": ["no time pressure", "needs attention soon", "blocking issue or hard deadline"],
        },
        "needs_reply": {
            "type": "noul",
            "instructions": "Does the sender expect a reply?",
        },
    }
