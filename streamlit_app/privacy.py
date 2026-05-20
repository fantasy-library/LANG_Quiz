"""PII detection and scrubbing for Canvas quiz CSV uploads."""

from __future__ import annotations

import re
from typing import Any

import pandas as pd

# Column name substrings that indicate PII (case-insensitive match on normalized name).
_PII_NAME_PATTERNS: tuple[str, ...] = (
    "name",
    "student name",
    "student_name",
    "full name",
    "id",
    "student id",
    "student_id",
    "sis_user_id",
    "sis_login_id",
    "login_id",
    "user_id",
    "email",
    "e-mail",
    "barcode",
    "section_id",
    "section_sis_id",
)

# Whole-column patterns scanned on string/object columns.
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[a-z]{2,}\b", re.IGNORECASE)
_HK_8DIGIT_RE = re.compile(r"\b\d{8}\b")
_HKUST_ID_RE = re.compile(r"\b[a-z]{2,4}\d{4,6}\b", re.IGNORECASE)

_PATTERN_LABELS: tuple[tuple[re.Pattern[str], str], ...] = (
    (_EMAIL_RE, "email"),
    (_HK_8DIGIT_RE, "hk_8digit_id"),
    (_HKUST_ID_RE, "hkust_id"),
)


def _column_matches_pii_name(col: str) -> bool:
    """Return True if column name matches a known PII name pattern."""
    col_lower = col.strip().lower()
    col_norm = col_lower.replace("_", " ")
    if col_lower in {"section", "score", "attempt", "submitted", "n correct", "n incorrect"}:
        return False
    exact = {p.lower().replace("_", " ") for p in _PII_NAME_PATTERNS}
    if col_norm in exact or col_lower in {p.lower() for p in _PII_NAME_PATTERNS}:
        return True
    for pattern in _PII_NAME_PATTERNS:
        p = pattern.lower()
        if len(p) <= 3:
            if col_norm == p or col_norm.endswith(f" {p}") or col_norm.startswith(f"{p} "):
                return True
        elif p in col_lower:
            return True
    return False


def _redact_series(series: pd.Series) -> tuple[pd.Series, list[str]]:
    """Redact PII pattern matches in a string column; return labels hit."""
    found: list[str] = []
    if series.dtype != object and not pd.api.types.is_string_dtype(series):
        return series, found

    def _redact_cell(val: object) -> object:
        if pd.isna(val):
            return val
        text = str(val)
        for pattern, label in _PATTERN_LABELS:
            if pattern.search(text):
                if label not in found:
                    found.append(label)
                text = pattern.sub("[REDACTED]", text)
        return text

    return series.map(_redact_cell), found


def _scan_column_for_patterns(series: pd.Series) -> list[str]:
    """Return pattern labels found in a string column (pre-redaction check)."""
    found: list[str] = []
    if series.dtype != object and not pd.api.types.is_string_dtype(series):
        return found
    text = series.dropna().astype(str)
    if text.empty:
        return found
    combined = " ".join(text.head(500).tolist())
    for pattern, label in _PATTERN_LABELS:
        if pattern.search(combined) and label not in found:
            found.append(label)
    return found


def scrub_pii(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Remove PII from a raw quiz DataFrame.

    This must be the first processing step after CSV read. Deterministic and
    side-effect free: returns a new DataFrame and an audit report dict.

    Parameters
    ----------
    df : pd.DataFrame
        Raw data as read from CSV (before any other processing).

    Returns
    -------
    tuple[pd.DataFrame, dict]
        Scrubbed DataFrame with anonymous index labels, and
        ``pii_report`` with keys ``columns_dropped`` and ``patterns_redacted``.
    """
    out = df.copy()
    columns_dropped: list[str] = []
    patterns_redacted: list[str] = []

    # 1. Drop columns by known PII name patterns
    to_drop_name: list[str] = []
    for col in out.columns:
        if _column_matches_pii_name(str(col)):
            to_drop_name.append(col)
    if to_drop_name:
        out = out.drop(columns=to_drop_name, errors="ignore")
        columns_dropped.extend(to_drop_name)

    # 2. Scan remaining string columns; redact in-cell PII (preserve quiz columns)
    for col in out.columns:
        series = out[col]
        if series.dtype != object and not pd.api.types.is_string_dtype(series):
            continue
        hits = _scan_column_for_patterns(series)
        if hits:
            redacted, _ = _redact_series(series)
            out[col] = redacted
            for h in hits:
                entry = f"{col} ({h})"
                if entry not in patterns_redacted:
                    patterns_redacted.append(entry)

    # 3. Anonymous row labels
    n = len(out)
    out.index = [f"Student {i + 1}" for i in range(n)]
    out.index.name = "Anonymous ID"

    pii_report: dict[str, Any] = {
        "columns_dropped": sorted(set(columns_dropped)),
        "patterns_redacted": patterns_redacted,
    }
    return out, pii_report
