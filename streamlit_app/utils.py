"""Data loading, parsing, and filter helpers for the quiz dashboard."""

from __future__ import annotations

import collections
import io
import random
import re
import string
from typing import Any, BinaryIO

import pandas as pd

from privacy import scrub_pii

_SCORE_COL_RE = re.compile(r"^column\s*(\d+)(?:\.\d+)?$", re.IGNORECASE)
_METADATA_COLUMNS = frozenset(
    {
        "section",
        "submitted",
        "attempt",
        "score",
        "n correct",
        "n incorrect",
        "anonymous id",
    }
)
_OPEN_ENDED_HINTS = ("most useful", "learnt", "learned", "comment", "suggestion", "short repl")
_MCQ_OPTION_SPLIT_RE = re.compile(r",\s*(?=[A-Z]\.)")
_MCQ_LETTER_RE = re.compile(r"^([A-Z])\.")
_PASS_THRESHOLD = 60.0
_WORD_RE = re.compile(r"[a-z][a-z0-9']{1,}")
_CANVAS_ARTIFACT_RE = re.compile(
    r"Links to an external site\.|\[[^\]]+\]\([^)]+\)|https?://\S+",
    re.IGNORECASE,
)
# Canvas question-bank IDs sometimes prefix exported headers (e.g. "1013201: Scenario...").
_CANVAS_QID_PREFIX_RE = re.compile(r"^\d{6,10}:\s*")


def clean_canvas_question_text(text: str) -> str:
    """Strip leading Canvas question ID from exported question header text."""
    return _CANVAS_QID_PREFIX_RE.sub("", str(text).strip(), count=1)
_EMPTY_RESPONSES = frozenset(
    {
        "",
        "no",
        "none",
        "n/a",
        "na",
        "-",
        "_",
        "no comment",
        "no comments",
        "nothing",
        "nil",
        "nope",
    }
)
_STOPWORDS = frozenset(
    """
    a an the and or but if in on at to for of is are was were be been being
    it its this that these those i me my we our you your he she they them
    as with from by about into through during before after above below
    not no nor so than too very can will just don should now have has had
    do does did doing would could should may might must shall will
    am is are was were be been being have has had having
    what which who whom whose when where why how
    all any both each few more most other some such no nor not only own same
    than too very s t can will just don should now
    there here then once
    """.split()
)
# Curated teaching topics: each label may match several spellings / aliases.
# Extend this list per course; no need to change grouping logic.
_CURATED_THEMES: tuple[dict[str, Any], ...] = (
    {"label": "PowerSearch", "needles": ("powersearch", "power search")},
    {"label": "Factiva", "needles": ("factiva",)},
    {"label": "ProQuest", "needles": ("proquest", "pro quest")},
    {"label": "EBSCOhost", "needles": ("ebsco", "ebscohost")},
    {"label": "Statista", "needles": ("statista",)},
    {"label": "HKEXnews", "needles": ("hkexnews", "hkex news")},
    {"label": "Passport database", "needles": ("passport",)},
    {
        "label": "Citation / referencing",
        "needles": (
            "citation",
            "citations",
            "reference list",
            "referencing",
            "reference",
            "references",
            "bibliography",
            "works cited",
            "generate a reference",
        ),
    },
    {"label": "APA style", "needles": ("apa", "apa style", "apa format", "apa 7")},
    {"label": "Databases", "needles": ("database", "databases")},
    {
        "label": "Research skills",
        "needles": (
            "research skills",
            "research skill",
            "research effectively",
            "research method",
            "finding sources",
            "find sources",
            "search for information",
            "filter search",
        ),
    },
    {"label": "Plagiarism / integrity", "needles": ("plagiarism", "academic integrity", "academic crime", "chatgpt", "paraphras")},
    {
        "label": "Evaluating sources",
        "needles": (
            "evaluating",
            "evaluation criteria",
            "crap",
            "currency",
            "relevancy",
            "check the source",
            "verify the source",
            "sources before",
            "credible",
            "credibility",
            "reliable source",
            "accurate source",
        ),
    },
    {"label": "Peer-reviewed sources", "needles": ("peer-reviewed", "peer reviewed", "peer review", "scholarly source")},
    {
        "label": "Library resources",
        "needles": (
            "libguide",
            "libguides",
            "library resources",
            "library website",
            "library databases",
            "library skills",
            "hkust library",
            "search databases",
        ),
    },
    {
        "label": "Citing sources",
        "needles": (
            "citing",
            "in-text citation",
            "in-text",
            "cite",
            "cited",
            "cites",
            "how to cite",
            "scholarly cite",
            "scholarlycite",
            "properly cite",
        ),
    },
    {"label": "Google / Google Scholar", "needles": ("google scholar", "google")},
    {"label": "Videos", "needles": ("video", "videos")},
    {"label": "Interactive exercises", "needles": ("interactive exercise", "interactive exercises", "interactive")},
    {
        "label": "Copyright / licensing",
        "needles": ("copyright", "license", "licenses", "licensing", "creative commons"),
    },
)

# Generic words that are not useful as auto-discovered topic labels.
_GENERIC_DISCOVERY_BLOCKLIST = frozenset(
    """
    useful learnt learned learning learn help helps helped helpful
    taught teach teaches teaching teacher teachers
    thing things module modules online assignment assignments
    form
    student students university hkust ust quiz answer answers question questions
    read reading write writing future academic work working
    good great nice well better best bad worse
    really actually basically definitely probably maybe perhaps
    know knew known think thinking thought feel feeling feelings felt
    like still even though however therefore because since
    use used using uses user find finding found search searching searched
    source sources resource resources information info
    cite citing cited citation different differently
    make making made understand understanding understood clear clearly
    easy easier hard harder simple simply
    way ways part parts kind kinds type types
    get got getting give gave given take took taken
    need needs needed want wants wanted try tried trying
    say said says tell told shows show showed
    see saw seen look looked looking
    short reply replies following result results differently
    lang lang1406 guide guides exercise exercises
    quite instead properly believe pretty effectively rather provided
    provide provides
    always often sometimes usually already always
    instead rather
  something anything everything someone anyone
  lot lots bit thank thanks okay ok yes sure able
  also time times day days year years
    """.split()
)

# Frequent reflection / feedback words that are not library topics (never auto-theme).
_DISCOVERY_FILLER_TOKENS = frozenset(
    """
    various variety avoid avoids avoided careful carefully correct correctly
    tools tool articles article websites website professional professionals
    access accessed ensure ensures evaluate evaluates evaluated evaluation
    relevant relevance important importance studies study start starts started
    appropriate appropriately examples example links link informative include
    includes included content contents follow follows suggest suggests shorter
    perfect concise comprehensive enough section sections games game
    instructions instruction their business school schools essays essay
    carefulness careful avoidable
    """.split()
)

# Used by word-frequency charts and theme discovery (English + open-ended filler).
_ANALYSIS_STOPWORDS = _STOPWORDS | _GENERIC_DISCOVERY_BLOCKLIST | _DISCOVERY_FILLER_TOKENS

# Tokens to never promote as auto-discovered themes (too generic or already curated).
_DISCOVERY_BLOCKLIST = _ANALYSIS_STOPWORDS

_DISCOVERY_MIN_RESPONSES = 3
_DISCOVERY_TOP_N = 10
_DISCOVERY_MIN_TOKEN_LEN = 5


_SCORE_COLUMN_ALIASES: tuple[str, ...] = (
    "score",
    "final score",
    "quiz score",
    "current score",
    "total score",
    "percent",
)


def _find_overall_score_column(df: pd.DataFrame) -> str | None:
    """Return the source column name for overall quiz score, if present."""
    lower_to_col = {str(c).strip().lower(): c for c in df.columns}
    for alias in _SCORE_COLUMN_ALIASES:
        if alias in lower_to_col:
            return lower_to_col[alias]
    return None


def ensure_score_column(df: pd.DataFrame, questions_meta: list[dict[str, Any]]) -> pd.DataFrame:
    """
    Ensure a numeric ``score`` column (%).

    Uses the Canvas export column when present; otherwise sums per-question points.
    """
    out = df.copy()
    src = _find_overall_score_column(out)
    if src is not None:
        out["score"] = pd.to_numeric(out[src], errors="coerce")
        if out["score"].notna().any():
            return out
    if not questions_meta:
        return out
    max_total = sum(float(q["max_score"]) for q in questions_meta)
    if max_total <= 0:
        return out
    total = pd.Series(0.0, index=out.index, dtype=float)
    for q in questions_meta:
        col = q["score_col"]
        if col not in out.columns:
            continue
        total = total + pd.to_numeric(out[col], errors="coerce").fillna(0)
    out["score"] = (total / max_total) * 100.0
    return out


def _extract_section_from_sis(df: pd.DataFrame) -> pd.DataFrame:
    """Populate ``section`` from ``section_sis_id`` before PII scrubbing."""
    out = df.copy()
    if "section_sis_id" in out.columns:
        out["section"] = (
            out["section_sis_id"]
            .astype(str)
            .str.split("-")
            .str[-1]
            .str.strip()
        )
    elif "section" not in out.columns:
        out["section"] = "Unknown"
    return out


def _normalize_column_name(name: object) -> str:
    """Strip whitespace and UTF-8 BOM from a CSV header."""
    return str(name).strip().lstrip("\ufeff")


def _is_metadata_column(name: object) -> bool:
    low = _normalize_column_name(name).lower()
    if low in _METADATA_COLUMNS:
        return True
    return low.startswith("section_") or low.startswith("unnamed")


def _is_score_column_name(name: object) -> bool:
    return bool(_SCORE_COL_RE.match(_normalize_column_name(name)))


def _looks_like_numeric_score_column(series: pd.Series) -> bool:
    """True when a column mostly holds numeric quiz points (0-100)."""
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().sum() < max(1, len(series) // 10):
        return False
    vals = numeric.dropna()
    if vals.empty:
        return True
    return float(vals.max()) <= 100.0


def _is_open_ended_pair(question_col: object, score_col: str, series: pd.Series) -> bool:
    """Classify ungraded / reflection prompts (max score 0)."""
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().any() and float(numeric.max()) > 0.0:
        return False
    q_low = _normalize_column_name(question_col).lower()
    if any(hint in q_low for hint in _OPEN_ENDED_HINTS):
        return True
    if not numeric.notna().any():
        return True
    return float(numeric.max()) <= 0.0


def _append_question_pair(
    *,
    scored: list[dict[str, Any]],
    open_ended: list[dict[str, Any]],
    q_num: int,
    o_num: int,
    question_col: object,
    score_col: str,
    series: pd.Series,
) -> tuple[int, int]:
    """Add one scored or open-ended question entry; return updated counters."""
    q_text = clean_canvas_question_text(str(question_col))
    if _is_open_ended_pair(question_col, score_col, series):
        o_num += 1
        open_ended.append(
            {
                "label": _open_ended_short_label(q_text, o_num),
                "question": q_text,
                "question_col": question_col,
                "score_col": score_col,
            }
        )
        return q_num, o_num
    q_num += 1
    max_score = float(pd.to_numeric(series, errors="coerce").max())
    scored.append(
        {
            "q_label": f"Q{q_num}",
            "question": q_text,
            "question_col": question_col,
            "score_col": score_col,
            "max_score": max_score,
        }
    )
    return q_num, o_num


def diagnose_quiz_columns(df: pd.DataFrame) -> str:
    """Human-readable hint when question pairing fails."""
    cols = [_normalize_column_name(c) for c in df.columns]
    score_like = [c for c in cols if _is_score_column_name(c)]
    lines = [
        f"Found {len(cols)} columns, {len(score_like)} score-like headers "
        f"(e.g. Column10, Column12).",
    ]
    if not score_like:
        samples = ", ".join(repr(c[:40]) for c in cols[:8])
        lines.append(
            "No ColumnNN headers detected. Use Canvas: Quiz Statistics -> "
            "Student Analysis -> Download CSV (not Item Analysis or Gradebook export)."
        )
        lines.append(f"First columns: {samples}")
    return " ".join(lines)


def _open_ended_short_label(question_text: str, index: int) -> str:
    """Derive a short UI label from the open-ended prompt text."""
    q = question_text.lower()
    if "most useful" in q or "learnt" in q or "learned" in q:
        return "Reflection"
    if "comment" in q or "suggestion" in q:
        return "Comments & suggestions"
    return f"Open-ended {index}"


def _scan_question_pairs(df: pd.DataFrame) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Pair question columns with score columns; split scored vs open-ended.

    Expects Canvas Student Analysis CSV: question text header, then ColumnNN points.
    Falls back to numeric columns that follow long text headers when ColumnNN is missing.
    """
    cols = list(df.columns)
    scored: list[dict[str, Any]] = []
    open_ended: list[dict[str, Any]] = []
    q_num = 0
    o_num = 0
    paired_score_cols: set[object] = set()

    for i, col in enumerate(cols):
        if not _is_score_column_name(col):
            continue
        if i == 0:
            continue
        question_col = cols[i - 1]
        if _is_metadata_column(question_col):
            continue
        score_col = col
        paired_score_cols.add(score_col)
        series = df[score_col]
        q_num, o_num = _append_question_pair(
            scored=scored,
            open_ended=open_ended,
            q_num=q_num,
            o_num=o_num,
            question_col=question_col,
            score_col=score_col,
            series=series,
        )

    for i in range(len(cols) - 1):
        question_col = cols[i]
        score_col = cols[i + 1]
        if score_col in paired_score_cols:
            continue
        if _is_metadata_column(question_col) or _is_metadata_column(score_col):
            continue
        if _is_score_column_name(score_col):
            continue
        if len(_normalize_column_name(question_col)) < 12:
            continue
        if not _looks_like_numeric_score_column(df[score_col]):
            continue
        paired_score_cols.add(score_col)
        q_num, o_num = _append_question_pair(
            scored=scored,
            open_ended=open_ended,
            q_num=q_num,
            o_num=o_num,
            question_col=question_col,
            score_col=score_col,
            series=df[score_col],
        )

    return scored, open_ended


def _identify_question_pairs(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Return scored multiple-choice question metadata only."""
    scored, _ = _scan_question_pairs(df)
    return scored


def _identify_open_ended_pairs(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Return open-ended (ungraded) question metadata."""
    _, open_ended = _scan_question_pairs(df)
    return open_ended


def normalize_open_text(raw: object) -> str:
    """Clean a single open-ended response for analysis and display."""
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return ""
    text = str(raw).strip()
    if not text:
        return ""
    text = _CANVAS_ARTIFACT_RE.sub(" ", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{2,}", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def is_meaningful_response(text: str) -> bool:
    """True if text counts as a non-empty student response."""
    cleaned = normalize_open_text(text).lower().strip(string.punctuation + " ")
    return cleaned not in _EMPTY_RESPONSES and len(cleaned) >= 3


def tokenize_for_analysis(text: str) -> list[str]:
    """Extract lowercase word tokens; drops stopwords and generic open-ended filler."""
    text = normalize_open_text(text).lower()
    return [w for w in _WORD_RE.findall(text) if w not in _ANALYSIS_STOPWORDS and len(w) > 2]


def _normalized_compact(text: str) -> tuple[str, str]:
    """Return lowercase text and a space-stripped variant for fuzzy matching."""
    lower = normalize_open_text(text).lower()
    return lower, lower.replace(" ", "")


_CITE_FORMS_RE = re.compile(r"\bcit(?:e|es|ed|ing)\b", re.IGNORECASE)


def response_matches_needles(text: str, needles: tuple[str, ...]) -> bool:
    """True if any alias appears in the response (handles spacing variants)."""
    normalized = normalize_open_text(text)
    lower, compact = _normalized_compact(text)
    for needle in needles:
        n = needle.lower().strip()
        if not n:
            continue
        if n == "cite":
            if _CITE_FORMS_RE.search(normalized):
                return True
            continue
        if " " not in n and len(n) <= 6:
            if response_matches_word(normalized, n):
                return True
        if n in lower or n.replace(" ", "") in compact:
            return True
    return False


def _effective_needles_for_prompt(
    needles: tuple[str, ...],
    prompt_text: str | None,
) -> tuple[str, ...]:
    """Drop phrases that appear in the question prompt (students echo the wording)."""
    if not prompt_text:
        return needles
    prompt_lower, prompt_compact = _normalized_compact(prompt_text)
    kept: list[str] = []
    for needle in needles:
        n = needle.lower().strip()
        if not n:
            continue
        if n in prompt_lower or n.replace(" ", "") in prompt_compact:
            continue
        kept.append(needle)
    return tuple(kept)


def response_matches_word(text: str, word: str) -> bool:
    """True if ``word`` appears as a whole token in the response."""
    pattern = re.compile(rf"\b{re.escape(word.lower())}\b", re.IGNORECASE)
    return bool(pattern.search(normalize_open_text(text)))


def _all_curated_needles() -> frozenset[str]:
    needles: set[str] = set()
    for theme in _CURATED_THEMES:
        needles.update(n.lower() for n in theme["needles"])
    return frozenset(needles)


def _discovery_blocklist_for_prompt(prompt_text: str | None) -> frozenset[str]:
    """Blocklist for auto-discovery, including words copied from the assignment prompt."""
    blocklist = set(_DISCOVERY_BLOCKLIST)
    if prompt_text:
        blocklist.update(tokenize_for_analysis(prompt_text))
    return frozenset(blocklist)


def _discovered_label(token: str) -> str:
    """Human-readable label for a single-token discovered theme."""
    return token[0].upper() + token[1:] if token else token


def _is_valid_discovered_token(
    token: str,
    blocklist: frozenset[str],
    curated: frozenset[str],
) -> bool:
    if len(token) < _DISCOVERY_MIN_TOKEN_LEN:
        return False
    if token in _DISCOVERY_FILLER_TOKENS:
        return False
    if len(token) >= 6 and token.endswith("ly"):
        return False
    if token in blocklist or token in curated:
        return False
    if any(token in c or c in token for c in curated if len(c) > 3):
        return False
    return True


def discover_prominent_terms(
    responses_df: pd.DataFrame,
    min_responses: int | None = None,
    top_n: int = _DISCOVERY_TOP_N,
    prompt_text: str | None = None,
) -> list[dict[str, Any]]:
    """
    Find frequent words in student text that are not covered by curated themes.

    Skips prompt boilerplate, stopwords, and short/generic tokens so labels stay meaningful.
    Returns dicts with keys ``label``, ``needles`` (single token), ``kind`` = discovered.
    """
    if responses_df.empty:
        return []
    n_resp = len(responses_df)
    threshold = min_responses if min_responses is not None else max(
        _DISCOVERY_MIN_RESPONSES, int(n_resp * 0.01)
    )
    curated = _all_curated_needles()
    blocklist = _discovery_blocklist_for_prompt(prompt_text)
    doc_freq: collections.Counter[str] = collections.Counter()

    for text in responses_df["Response"]:
        seen = set(tokenize_for_analysis(text))
        for token in seen:
            doc_freq[token] += 1

    discovered: list[dict[str, Any]] = []
    for token, count in doc_freq.most_common():
        if count < threshold:
            break
        if not _is_valid_discovered_token(token, blocklist, curated):
            continue
        discovered.append(
            {
                "label": _discovered_label(token),
                "needles": (token,),
                "kind": "discovered",
                "doc_count": count,
            }
        )
        if len(discovered) >= top_n:
            break
    return discovered


def build_theme_catalog(
    responses_df: pd.DataFrame,
    prompt_text: str | None = None,
    *,
    include_discovered: bool = False,
) -> list[dict[str, Any]]:
    """Curated themes; optionally add frequent words not already covered."""
    catalog: list[dict[str, Any]] = []
    for theme in _CURATED_THEMES:
        catalog.append({**theme, "kind": "curated"})
    if not include_discovered:
        return catalog
    seen_labels = {t["label"] for t in catalog}
    for item in discover_prominent_terms(responses_df, prompt_text=prompt_text):
        if item["label"] not in seen_labels:
            catalog.append(item)
            seen_labels.add(item["label"])
    return catalog


def sort_theme_counts_for_display(
    theme_counts: dict[str, int],
    theme_kinds: dict[str, str] | None = None,
) -> list[tuple[str, int]]:
    """Curated topics first, then discovered; both sorted by response count."""
    kinds = theme_kinds or {}

    def sort_key(item: tuple[str, int]) -> tuple[int, int, str]:
        label, count = item
        curated_rank = 0 if kinds.get(label) == "curated" else 1
        return (curated_rank, -count, label.lower())

    return sorted(theme_counts.items(), key=sort_key)


# Second-pass matchers for responses still in Other (prompt-filtered needles miss many short replies).
_OTHER_REASSIGN_ORDER: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("PowerSearch", ("powersearch", "power search")),
    ("ProQuest", ("proquest", "pro quest")),
    ("EBSCOhost", ("ebsco", "ebscohost")),
    ("Google / Google Scholar", ("google scholar", "google")),
    ("APA style", ("apa", "apa style", "apa format")),
    ("Plagiarism / integrity", ("plagiarism", "academic integrity", "chatgpt", "paraphras")),
    (
        "Citing sources",
        (
            "cite",
            "citing",
            "in-text citation",
            "how to cite",
            "scholarlycite",
            "scholarly cite",
            "properly cite",
        ),
    ),
    (
        "Citation / referencing",
        (
            "citation",
            "citations",
            "reference list",
            "referencing",
            "reference",
            "references",
            "bibliography",
            "works cited",
            "generate a reference",
        ),
    ),
    (
        "Evaluating sources",
        (
            "check the source",
            "verify the source",
            "sources before",
            "credible",
            "credibility",
            "reliable source",
            "accurate source",
            "evaluation criteria",
            "currency",
            "relevancy",
        ),
    ),
    ("Peer-reviewed sources", ("peer-reviewed", "peer reviewed", "scholarly source")),
    (
        "Library resources",
        (
            "libguide",
            "library databases",
            "library website",
            "library skills",
            "hkust library",
            "search databases",
            "search widely",
            "search smart",
            "using the library",
        ),
    ),
    (
        "Research skills",
        (
            "research effectively",
            "research skills",
            "finding sources",
            "find sources",
            "search for information",
            "filter search",
            "research method",
            "search efficiently",
            "search effectively",
            "search engine",
            "conduct research",
            "during research",
        ),
    ),
    ("Databases", ("database", "databases", "data site", "data sites")),
    ("Statista", ("statista",)),
    ("Factiva", ("factiva",)),
    ("Passport database", ("passport",)),
    ("HKEXnews", ("hkexnews",)),
    ("Videos", ("video", "videos")),
    ("Interactive exercises", ("interactive exercise", "interactive")),
)


def _match_secondary_theme(text: str) -> str | None:
    """Pick the first matching curated topic for a formerly uncategorized response."""
    for label, needles in _OTHER_REASSIGN_ORDER:
        if response_matches_needles(text, needles):
            return label
    return None


def _reassign_other_responses(
    grouped: dict[str, pd.DataFrame],
    theme_kind: dict[str, str],
) -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    """Move Other responses into curated topics when secondary needles match."""
    other_key = "Other / uncategorized"
    if other_key not in grouped:
        return grouped, theme_kind
    other_df = grouped.pop(other_key)
    if other_df.empty:
        return grouped, theme_kind

    remaining_rows: list[pd.Series] = []
    for _, row in other_df.iterrows():
        label = _match_secondary_theme(str(row["Response"]))
        if label is None:
            remaining_rows.append(row)
            continue
        piece = row.to_frame().T
        if label in grouped:
            grouped[label] = pd.concat([grouped[label], piece], ignore_index=True)
        else:
            grouped[label] = piece
        theme_kind[label] = theme_kind.get(label, "curated")

    if remaining_rows:
        grouped[other_key] = pd.DataFrame(remaining_rows).reset_index(drop=True)
        theme_kind[other_key] = "other"
    return grouped, theme_kind


def group_responses_by_theme(
    responses_df: pd.DataFrame,
    prompt_text: str | None = None,
) -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    """
    Group responses under curated and discovered theme labels.

    Returns ``(grouped_dfs, theme_kind)`` where ``theme_kind[label]`` is
    ``curated``, ``discovered``, or ``other``.
    """
    if responses_df.empty:
        return {}, {}
    catalog = build_theme_catalog(responses_df, prompt_text=prompt_text)
    grouped: dict[str, pd.DataFrame] = {}
    theme_kind: dict[str, str] = {}
    matched_ids: set[str] = set()

    for theme in catalog:
        label = theme["label"]
        needles = tuple(theme["needles"])
        if theme.get("kind") != "discovered":
            needles = _effective_needles_for_prompt(needles, prompt_text)
            if not needles:
                continue
        if theme.get("kind") == "discovered":
            mask = responses_df["Response"].map(
                lambda t, w=needles[0]: response_matches_word(t, w)
            )
        else:
            mask = responses_df["Response"].map(
                lambda t, ns=needles: response_matches_needles(t, ns)
            )
        sub = responses_df[mask].copy()
        if sub.empty:
            continue
        grouped[label] = sub
        theme_kind[label] = theme.get("kind", "curated")
        matched_ids.update(sub["Anonymous ID"].astype(str).tolist())

    other = responses_df[~responses_df["Anonymous ID"].astype(str).isin(matched_ids)]
    if not other.empty:
        grouped["Other / uncategorized"] = other
        theme_kind["Other / uncategorized"] = "other"
    grouped, theme_kind = _reassign_other_responses(grouped, theme_kind)
    return grouped, theme_kind


def count_themes_in_responses(
    responses_df: pd.DataFrame,
    prompt_text: str | None = None,
) -> dict[str, int]:
    """Count how many responses mention each theme label."""
    if responses_df.empty:
        return {}
    counts: dict[str, int] = {}
    for theme in build_theme_catalog(responses_df, prompt_text=prompt_text):
        label = theme["label"]
        needles = tuple(theme["needles"])
        if theme.get("kind") != "discovered":
            needles = _effective_needles_for_prompt(needles, prompt_text)
            if not needles:
                continue
        n = 0
        for text in responses_df["Response"]:
            if theme.get("kind") == "discovered":
                if response_matches_word(text, needles[0]):
                    n += 1
            elif response_matches_needles(text, needles):
                n += 1
        if n > 0:
            counts[label] = n
    return counts


def collect_open_responses(df: pd.DataFrame, question_col: str) -> pd.DataFrame:
    """Return anonymised id and full cleaned text for every meaningful response."""
    rows: list[dict[str, str]] = []
    if question_col not in df.columns:
        return pd.DataFrame(columns=["Anonymous ID", "Response"])
    for anon_id, raw in df[question_col].items():
        text = normalize_open_text(raw)
        if is_meaningful_response(text):
            rows.append({"Anonymous ID": str(anon_id), "Response": text})
    return pd.DataFrame(rows)


def analyze_open_text(
    series: pd.Series,
    sample_seed: int = 42,
    prompt_text: str | None = None,
) -> dict[str, Any]:
    """
    Summarize open-ended responses with stdlib tools (``re``, ``collections``).

    Returns response rates, length stats, top terms, theme hits, and samples.
    """
    normalized = series.map(normalize_open_text)
    meaningful = normalized[normalized.map(is_meaningful_response)]
    n_total = len(series)
    n_answered = len(meaningful)
    word_counter: collections.Counter[str] = collections.Counter()
    char_lengths: list[int] = []
    word_lengths: list[int] = []

    for text in meaningful:
        tokens = tokenize_for_analysis(text)
        word_counter.update(tokens)
        char_lengths.append(len(text))
        word_lengths.append(len(tokens))

    meaningful_df = pd.DataFrame({"Response": meaningful.tolist()})
    theme_counts = count_themes_in_responses(meaningful_df, prompt_text=prompt_text)
    catalog = build_theme_catalog(meaningful_df, prompt_text=prompt_text)
    theme_kinds = {
        t["label"]: t.get("kind", "curated")
        for t in catalog
        if t["label"] in theme_counts
    }

    samples: list[str] = []
    if len(meaningful) > 0:
        sample_n = min(8, len(meaningful))
        rng = random.Random(sample_seed)
        samples = rng.sample(meaningful.tolist(), sample_n)

    return {
        "n_total": n_total,
        "n_answered": n_answered,
        "response_rate_pct": (n_answered / n_total * 100) if n_total else 0.0,
        "median_chars": float(pd.Series(char_lengths).median()) if char_lengths else 0.0,
        "median_words": float(pd.Series(word_lengths).median()) if word_lengths else 0.0,
        "top_words": word_counter.most_common(20),
        "theme_counts": theme_counts,
        "theme_kinds": theme_kinds,
        "samples": samples,
        "char_lengths": char_lengths,
        "word_lengths": word_lengths,
    }


# Canvas quiz exports are usually Windows-1252 / cp1252, but bytes like 0x9d are
# undefined in cp1252 and break UTF-8. latin-1 accepts every byte and never fails.


def _bytes_are_valid_utf8(data: bytes) -> bool:
    """True only when the file is real UTF-8 (not cp1252 mis-detected)."""
    try:
        data.decode("utf-8")
        return True
    except UnicodeDecodeError:
        return False


def _coerce_csv_bytes(source: bytes | bytearray | BinaryIO | io.BytesIO) -> bytes:
    """Read the full upload as raw bytes (no UTF-8 text decode)."""
    if isinstance(source, (bytes, bytearray)):
        return bytes(source)
    if hasattr(source, "getvalue") and callable(source.getvalue):
        return bytes(source.getvalue())
    if hasattr(source, "read") and callable(source.read):
        return bytes(source.read())
    raise TypeError(f"Expected bytes or a binary stream, got {type(source).__name__}")


def _read_canvas_csv_bytes(data: bytes) -> pd.DataFrame:
    """Parse Canvas CSV bytes without UTF-8/cp1252 decode failures."""
    if not data:
        raise ValueError("CSV file is empty.")
    # Use UTF-8 only when the whole file is valid UTF-8 (chardet often guesses wrong).
    if _bytes_are_valid_utf8(data):
        try:
            return pd.read_csv(io.BytesIO(data), encoding="utf-8-sig", engine="c")
        except Exception:
            pass
    # latin-1 maps every byte 0x00-0xFF; safe for Windows Canvas / Excel exports.
    return pd.read_csv(io.StringIO(data.decode("latin-1")), engine="c")


def load_and_parse(file: bytes | bytearray | BinaryIO | io.BytesIO) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]], list[str], dict[str, Any]]:
    """
    Read a Canvas quiz CSV, scrub PII, and build question metadata.

    Parameters
    ----------
    file : bytes or file-like
        Raw uploaded CSV bytes (not a decoded string).

    Returns
    -------
    tuple
        Scrubbed ``df``, scored ``questions_meta``, ``open_ended_meta``,
        sorted ``sections``, and ``pii_report`` from :func:`privacy.scrub_pii`.
    """
    raw = _read_canvas_csv_bytes(_coerce_csv_bytes(file))
    raw.columns = pd.Index([_normalize_column_name(c) for c in raw.columns])

    # Extract section label while section_sis_id is still present
    raw = _extract_section_from_sis(raw)

    # PII scrub - mandatory before any other processing
    df, pii_report = scrub_pii(raw)

    if "submitted" in df.columns:
        df["submitted"] = pd.to_datetime(df["submitted"], utc=True, errors="coerce")

    questions_meta, open_ended_meta = _scan_question_pairs(df)
    df = ensure_score_column(df, questions_meta)

    sections: list[str] = []
    if "section" in df.columns:
        sections = sorted(df["section"].dropna().astype(str).unique().tolist())

    return df, questions_meta, open_ended_meta, sections, pii_report


def shuffle_anonymous(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Return a shuffled copy with renumbered anonymous index labels."""
    shuffled = df.sample(frac=1, random_state=seed).copy()
    shuffled.index = [f"Student {i + 1}" for i in range(len(shuffled))]
    shuffled.index.name = "Anonymous ID"
    return shuffled


def apply_filters(
    df: pd.DataFrame,
    sections: list[str],
    score_range: tuple[float, float],
    date_range: tuple[pd.Timestamp, pd.Timestamp] | None,
    pass_filter: str,
) -> pd.DataFrame:
    """Apply sidebar filters to the scrubbed DataFrame."""
    if not sections:
        return df.iloc[0:0].copy()
    out = df.copy()
    if "section" in out.columns:
        out = out[out["section"].astype(str).isin(sections)]
    if "score" in out.columns:
        lo, hi = score_range
        out = out[out["score"].between(lo, hi)]
    if date_range and "submitted" in out.columns:
        start, end = date_range
        submitted = out["submitted"]
        out = out[(submitted >= start) & (submitted <= end)]
    if pass_filter == "Passed (\u226560)" and "score" in out.columns:
        out = out[out["score"] >= _PASS_THRESHOLD]
    elif pass_filter == "Failed (<60)" and "score" in out.columns:
        out = out[out["score"] < _PASS_THRESHOLD]
    return out


def truncate_label(text: str, max_len: int = 60) -> str:
    """Truncate long text for chart axis labels."""
    text = str(text).replace("\n", " ")
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def split_mcq_response(text: str) -> list[str]:
    """Split a Canvas MCQ / multi-select cell into one string per lettered option."""
    cleaned = str(text).strip()
    if not cleaned or cleaned.lower() in ("nan", "no answer"):
        return []
    parts = _MCQ_OPTION_SPLIT_RE.split(cleaned)
    return [p.strip() for p in parts if p.strip()]


def mcq_letters_from_response(text: str) -> set[str]:
    """Return option letters (A, B, C, ...) present in a response."""
    letters: set[str] = set()
    for part in split_mcq_response(text):
        match = _MCQ_LETTER_RE.match(part)
        if match:
            letters.add(match.group(1).upper())
    return letters


def looks_like_mcq_multiselect(answers: pd.Series) -> bool:
    """True when answers look like lettered Canvas MCQ options (possibly multi-select)."""
    sample = answers.dropna().astype(str).head(80)
    if sample.empty:
        return False
    matched = sum(
        1
        for text in sample
        if _MCQ_LETTER_RE.match(str(text).strip()) or _MCQ_OPTION_SPLIT_RE.search(str(text))
    )
    return matched >= max(3, int(len(sample) * 0.5))


def count_mcq_letter_selections(answers: pd.Series) -> pd.DataFrame:
    """
    Count how many students selected each letter at least once.

    Used when students may pick multiple options (Canvas joins them with commas).
    """
    letter_counts: collections.Counter[str] = collections.Counter()
    letter_text: dict[str, str] = {}
    for raw in answers:
        for letter in mcq_letters_from_response(raw):
            letter_counts[letter] += 1
        for part in split_mcq_response(raw):
            match = _MCQ_LETTER_RE.match(part)
            if not match:
                continue
            letter = match.group(1).upper()
            prev_text = letter_text.get(letter, "")
            if len(part) > len(prev_text):
                letter_text[letter] = part.strip()
    rows = [
        {
            "Letter": letter,
            "Count": letter_counts[letter],
            "Full answer text": letter_text.get(letter, letter),
        }
        for letter in sorted(letter_counts)
    ]
    return pd.DataFrame(rows)


def correct_mcq_letters(correct_answers: set[str]) -> set[str]:
    """Map full-text correct answer(s) to option letters."""
    letters: set[str] = set()
    for answer in correct_answers:
        letters.update(mcq_letters_from_response(answer))
    return letters


def detect_correct_answers(
    df: pd.DataFrame, question_col: str, score_col: str, max_score: float
) -> set[str]:
    """
    Infer correct answer text(s) as choices earning full credit on average.
    """
    work = df[[question_col, score_col]].copy()
    work[score_col] = pd.to_numeric(work[score_col], errors="coerce")
    work[question_col] = work[question_col].fillna("No answer").astype(str)
    grouped = work.groupby(question_col)[score_col].mean()
    correct = set()
    for answer, mean_score in grouped.items():
        if pd.notna(mean_score) and mean_score >= max_score - 1e-6:
            correct.add(answer)
    return correct


def export_scrubbed_csv(df: pd.DataFrame) -> bytes:
    """Re-scrub and return CSV bytes for download (in-memory only)."""
    to_export = df.copy()
    to_export.index.name = "Anonymous ID"
    scrubbed, _ = scrub_pii(to_export)
    return scrubbed.to_csv(index=True).encode("utf-8")


def format_pii_message(pii_report: dict[str, Any]) -> str:
    """Build a user-facing PII scrub audit message."""
    dropped = pii_report.get("columns_dropped") or []
    patterns = pii_report.get("patterns_redacted") or []
    parts = ["\u2705 PII scrubbed"]
    if dropped:
        parts.append(f": dropped columns [{', '.join(dropped)}]")
    else:
        parts.append(": no named PII columns found")
    if patterns:
        parts.append(f". Pattern hits redacted: [{', '.join(patterns)}]")
    else:
        parts.append(". No sensitive patterns detected in remaining columns.")
    return "".join(parts)
