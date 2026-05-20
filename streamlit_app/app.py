"""LANG Quiz Results - Streamlit dashboard."""



from __future__ import annotations



import hashlib
import html
import io

import re

from typing import Any



import pandas as pd

import plotly.express as px
import plotly.graph_objects as go

import streamlit as st



from utils import (

    analyze_open_text,

    apply_filters,

    collect_open_responses,

    correct_mcq_letters,

    count_mcq_letter_selections,

    detect_correct_answers,

    diagnose_quiz_columns,

    export_scrubbed_csv,

    looks_like_mcq_multiselect,

    group_responses_by_theme,

    load_and_parse,

    sort_theme_counts_for_display,

    normalize_open_text,

    shuffle_anonymous,

    truncate_label,

)



PASS_THRESHOLD = 60.0

COLOR_PASS = "#4CAF50"

COLOR_FAIL = "#FF6B6B"


def _plotly_chart(
    fig,
    *,
    use_container_width: bool = True,
    responsive: bool = True,
) -> None:
    """
    Render Plotly without Streamlit's PlotlyChart JS chunk (fails on some PaaS hosts).

    Uses Plotly CDN in an HTML component so charts load on Railway and similar platforms.
    """
    import plotly.io as pio

    layout_h = fig.layout.height
    extra_h = 50
    if fig.layout.legend and fig.layout.legend.y is not None:
        try:
            if float(fig.layout.legend.y) < 0:
                extra_h = 70
        except (TypeError, ValueError):
            pass
    height = int(layout_h) + extra_h if layout_h else 450
    html = pio.to_html(
        fig,
        full_html=False,
        include_plotlyjs="cdn",
        config={"displayModeBar": True, "responsive": responsive},
    )
    width_style = "width:100%;" if use_container_width else ""
    st.components.v1.html(
        f'<div style="{width_style}">{html}</div>',
        height=height,
        scrolling=False,
    )


def _section_color_map(sections: list[str] | pd.Series) -> dict[str, str]:
    """Muted blue shades per section (readable with many sections, not rainbow)."""
    ordered = sorted({str(s) for s in sections})
    n = len(ordered)
    if n == 0:
        return {}
    if n == 1:
        return {ordered[0]: "#4A6FA5"}
    positions = [0.38 + 0.48 * i / (n - 1) for i in range(n)]
    colors = px.colors.sample_colorscale("Blues", positions)
    return dict(zip(ordered, colors))


def _apply_section_legend(fig, n_sections: int) -> None:
    """Legend layout that keeps full section codes visible (e.g. T01, not truncated to T0)."""
    base = dict(
        itemclick=False,
        itemdoubleclick=False,
        title=dict(text="Section"),
    )
    if n_sections <= 8:
        fig.update_layout(
            legend={**base, "font": dict(size=11), "itemwidth": 42, "tracegroupgap": 0},
            margin=dict(r=100, t=60, b=60),
            height=480,
        )
        return
    if n_sections <= 12:
        fig.update_layout(
            legend={
                **base,
                "font": dict(size=11),
                "orientation": "h",
                "yanchor": "top",
                "y": -0.22,
                "xanchor": "center",
                "x": 0.5,
                "itemwidth": 48,
                "tracegroupgap": 2,
            },
            margin=dict(b=110, t=60, l=55, r=55),
            height=540,
        )
        return
    # Many sections: horizontal legends clip labels (e.g. T21 -> "T"). Use vertical.
    row_px = 17
    fig.update_layout(
        legend={
            **base,
            "font": dict(size=10),
            "orientation": "v",
            "yanchor": "top",
            "y": 1.0,
            "xanchor": "left",
            "x": 1.01,
            "tracegroupgap": 1,
            "bgcolor": "rgba(255,255,255,0.85)",
            "bordercolor": "#ddd",
            "borderwidth": 1,
        },
        margin=dict(r=72, t=60, b=60, l=55),
        height=max(520, 100 + n_sections * row_px),
    )





st.set_page_config(

    layout="wide",

    page_title="LANG Quiz Dashboard",

)





@st.cache_data(show_spinner=False)
def cached_load(
    file_hash: str,
    file_bytes: bytes,
) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]], list[str], dict[str, Any]]:
    """Parse uploaded CSV bytes with PII scrubbing (cached per file content)."""
    _ = file_hash
    return load_and_parse(file_bytes)





def _section_checkbox_key(section: str) -> str:
    return f"filter_sec_{section}"


def _init_section_filter_state(sections: list[str], *, select_all: bool) -> None:
    """Initialize section checkbox keys (call on new file or Reset / Select all)."""
    st.session_state["sel_sections"] = list(sections) if select_all else []
    want = set(st.session_state["sel_sections"])
    for sec in sections:
        st.session_state[_section_checkbox_key(sec)] = sec in want


def _sections_from_checkbox_state(sections: list[str]) -> list[str]:
    """Read current checkbox ticks without overwriting widget state."""
    return [
        sec
        for sec in sections
        if st.session_state.get(_section_checkbox_key(sec), False)
    ]


def _section_filter_sidebar(sections: list[str]) -> list[str]:
    """
    Section filter using checkboxes so zero sections can be selected reliably.

    Checkbox clicks update ``st.session_state`` directly; do not resync from
  ``sel_sections`` before rendering or ticks revert on the next rerun.
    """
    for sec in sections:
        key = _section_checkbox_key(sec)
        if key not in st.session_state:
            st.session_state[key] = True

    st.markdown("**Section**")
    with st.expander("Choose sections", expanded=True):
        b1, b2 = st.columns(2)
        with b1:
            if st.button("Select all", key="filter_sec_all", use_container_width=True):
                _init_section_filter_state(sections, select_all=True)
                st.rerun()
        with b2:
            if st.button("Unselect all", key="filter_sec_none", use_container_width=True):
                _init_section_filter_state(sections, select_all=False)
                st.rerun()

        for sec in sections:
            st.checkbox(sec, key=_section_checkbox_key(sec))

        selected = _sections_from_checkbox_state(sections)
        st.caption(f"{len(selected)} of {len(sections)} sections selected")

    st.session_state["sel_sections"] = selected
    if not selected:
        st.caption("No sections selected - no data will be shown.")
    return selected


def _default_filter_state(sections: list[str], df: pd.DataFrame) -> dict[str, Any]:

    score_min = float(df["score"].min()) if "score" in df.columns and len(df) else 0.0

    score_max = float(df["score"].max()) if "score" in df.columns and len(df) else 100.0

    if "submitted" in df.columns and df["submitted"].notna().any():

        dmin = df["submitted"].min().date()

        dmax = df["submitted"].max().date()

    else:

        dmin = dmax = pd.Timestamp.utcnow().date()

    return {

        "sections": sections,

        "score_range": (score_min, score_max),

        "date_range": (dmin, dmax),

        "pass_filter": "All students",

    }





def _filter_banner(active: dict[str, Any], defaults: dict[str, Any]) -> None:

    parts: list[str] = []

    if set(active.get("sections", [])) != set(defaults.get("sections", [])):

        if not active.get("sections"):

            parts.append("Sections: (none selected)")

        else:

            parts.append(f"Sections: {', '.join(active['sections'])}")

    if active.get("score_range") != defaults.get("score_range"):

        lo, hi = active["score_range"]

        parts.append(f"Score: {lo:.0f}-{hi:.0f}%")

    if active.get("date_range") != defaults.get("date_range"):

        d0, d1 = active["date_range"]

        parts.append(f"Dates: {d0} to {d1}")

    if active.get("pass_filter") != defaults.get("pass_filter"):

        parts.append(active["pass_filter"])

    if parts:

        st.info("**Active filters:** " + " | ".join(parts))





def _kpi_row(df: pd.DataFrame) -> None:

    n = len(df)

    if n == 0 or "score" not in df.columns:

        st.warning("No students match the current filters.")

        return

    mean_s = df["score"].mean()

    median_s = df["score"].median()

    pass_rate = (df["score"] >= PASS_THRESHOLD).mean() * 100

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total students", f"{n:,}")

    c2.metric("Mean score", f"{mean_s:.1f}%")

    c3.metric("Pass rate (>=60%)", f"{pass_rate:.1f}%")

    c4.metric("Median score", f"{median_s:.1f}%")





def _score_histogram(df: pd.DataFrame) -> None:

    if "score" not in df.columns or df.empty:

        return

    scores = df["score"].astype(float)

    passed = scores[scores >= PASS_THRESHOLD]

    failed = scores[scores < PASS_THRESHOLD]

    mean_s = scores.mean()

    # Shared 5-point bins with a boundary at 60% so pass/fail colors align with the threshold.
    score_bins = dict(start=0, end=100, size=5)

    fig = go.Figure()

    fig.add_trace(

        go.Histogram(

            x=failed,

            xbins=score_bins,

            name="Failed",

            marker_color=COLOR_FAIL,

            hovertemplate="Failed<br>Score: %{x}<br>Count: %{y}<extra></extra>",

        )

    )

    fig.add_trace(

        go.Histogram(

            x=passed,

            xbins=score_bins,

            name="Passed",

            marker_color=COLOR_PASS,

            hovertemplate="Passed<br>Score: %{x}<br>Count: %{y}<extra></extra>",

        )

    )

    fig.update_layout(

        barmode="stack",

        bargap=0.05,

        title="Score distribution",

        xaxis_title="Score (%)",

        yaxis_title="Count",

        showlegend=True,

    )

    fig.add_vline(x=mean_s, line_dash="dash", line_color="#1565C0", annotation_text="Mean")

    fig.add_vline(

        x=PASS_THRESHOLD,

        line_dash="dot",

        line_color="#333",

        annotation_text="Pass (60%)",

    )

    _apply_hover_layout(fig)

    _plotly_chart(fig, use_container_width=True)





def _score_by_section_box(df: pd.DataFrame) -> None:

    if "section" not in df.columns or "score" not in df.columns or df.empty:

        return

    fig = px.box(

        df,

        x="section",

        y="score",

        labels={"section": "Section", "score": "Score (%)"},

        title="Scores by section",

        color_discrete_sequence=px.colors.sequential.Blues[3:6],

    )

    fig.update_traces(marker_color="#1976D2", line_color="#1565C0")
    _finish_simple_hover(fig, "Section: %{x}<br>Score: %{y}<extra></extra>")

    _plotly_chart(fig, use_container_width=True)





def _submission_timeline(df: pd.DataFrame) -> None:

    if "submitted" not in df.columns or df["submitted"].isna().all():

        return

    daily = (

        df.dropna(subset=["submitted"])

        .assign(day=lambda d: d["submitted"].dt.floor("D"))

        .groupby("day")

        .size()

        .reset_index(name="submissions")

    )

    fig = px.line(

        daily,

        x="day",

        y="submissions",

        markers=True,

        labels={"day": "Date", "submissions": "Submissions"},

        title="Submissions per day",

        color_discrete_sequence=px.colors.sequential.Blues[4:5],

    )
    _finish_simple_hover(fig, "Date: %{x}<br>Submissions: %{y}<extra></extra>")

    _plotly_chart(fig, use_container_width=True)





def _question_full_marks_chart(df: pd.DataFrame, questions_meta: list[dict[str, Any]]) -> None:

    if not questions_meta or df.empty:

        return

    rows = []

    for q in questions_meta:

        scores = pd.to_numeric(df[q["score_col"]], errors="coerce")

        pct = (scores >= q["max_score"] - 1e-6).mean() * 100

        rows.append(
            {
                "Question": q["q_label"],
                "At max points %": pct,
                "question_text": q["question"],
            }
        )

    qdf = pd.DataFrame(rows).sort_values("At max points %", ascending=True)
    qdf["Question hover"] = _question_hover_series(qdf["question_text"])

    fig = px.bar(

        qdf,

        x="At max points %",

        y="Question",

        orientation="h",

        custom_data=["Question hover"],

        labels={"At max points %": "% at maximum points"},

        title="Students who earned all points on the question",

        color="At max points %",

        color_continuous_scale="Blues",

    )

    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    _finish_question_bar_hover(fig, "At max points (%)")

    _plotly_chart(fig, use_container_width=True)

    st.caption(
        "Each bar is the percentage of students (in your current filters) whose **Canvas score on "
        "that question equals the question maximum** - i.e. they received every point available "
        "for that item. Lowest bars are the hardest or most often missed. "
        "The chart on the right shows **average** points instead, which includes partial credit."
    )





def _question_avg_score_chart(df: pd.DataFrame, questions_meta: list[dict[str, Any]]) -> None:

    if not questions_meta or df.empty:

        return

    rows = []

    for q in questions_meta:

        scores = pd.to_numeric(df[q["score_col"]], errors="coerce")

        avg_ratio = scores.mean() / q["max_score"] if q["max_score"] else 0

        rows.append(

            {

                "Question": q["q_label"],

                "Avg ratio": avg_ratio * 100,

                "question_text": q["question"],

            }

        )

    qdf = pd.DataFrame(rows).sort_values("Avg ratio", ascending=True)
    qdf["Question hover"] = _question_hover_series(qdf["question_text"])

    fig = px.bar(

        qdf,

        x="Avg ratio",

        y="Question",

        orientation="h",

        custom_data=["Question hover"],

        labels={"Avg ratio": "Average score / max (%)"},

        title="Average points earned on the question (% of max)",

        color="Avg ratio",

        color_continuous_scale="Blues",

    )

    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    _finish_question_bar_hover(fig, "Avg (% of max)")

    _plotly_chart(fig, use_container_width=True)

    st.caption(
        "Mean question score divided by that question's maximum points. Students with zero or "
        "partial credit lower this bar even when some classmates earned all points (left chart)."
    )





def _question_detail(df: pd.DataFrame, questions_meta: list[dict[str, Any]]) -> None:

    if not questions_meta:

        st.info(
            "No scored questions detected in this CSV. "
            + diagnose_quiz_columns(df)
        )
        return

    labels = [f"{q['q_label']}: {truncate_label(q['question'])}" for q in questions_meta]

    idx = st.selectbox("Select a question", range(len(labels)), format_func=lambda i: labels[i])

    q = questions_meta[idx]

    st.markdown(f"**{q['q_label']} - full text**")

    st.write(q["question"])



    work = df[[q["question_col"], q["score_col"]]].copy()

    work[q["question_col"]] = work[q["question_col"]].fillna("No answer").astype(str)

    work[q["score_col"]] = pd.to_numeric(work[q["score_col"]], errors="coerce")

    answer_col = work[q["question_col"]]
    correct = detect_correct_answers(df, q["question_col"], q["score_col"], q["max_score"])

    if looks_like_mcq_multiselect(answer_col):
        st.caption(
            "This is a **multi-select** question (students may choose more than one option). "
            "The chart counts how many students selected each letter (A, B, C, ...), not every "
            "unique combination of choices."
        )
        counts = count_mcq_letter_selections(answer_col)
        correct_letters = correct_mcq_letters(correct)
        counts["is_correct"] = counts["Letter"].isin(correct_letters)
        counts["Choice"] = counts["Letter"]
        counts["Answer hover"] = counts["Full answer text"].map(
            lambda t: _wrap_for_hover(str(t).replace("\n", " "), width=48)
        )
        chart_title = "Selections per option (multi-select MCQ)"
        legend_cols = ["Letter", "Count", "Correct", "Full answer text"]
    else:
        counts = answer_col.value_counts().reset_index()
        counts.columns = ["Answer", "Count"]
        counts["is_correct"] = counts["Answer"].isin(correct)
        counts = counts.sort_values("Count", ascending=True).reset_index(drop=True)
        counts["Choice"] = [f"Option {i + 1}" for i in range(len(counts))]
        counts["Answer hover"] = counts["Answer"].map(
            lambda t: _wrap_for_hover(str(t).replace("\n", " "), width=48)
        )
        chart_title = "Answer distribution (hover for full answer text)"
        legend_cols = ["Choice", "Count", "Correct", "Full answer text"]
        counts = counts.rename(columns={"Answer": "Full answer text"})

    counts["answer_key"] = counts["is_correct"].map(
        {True: "Correct", False: "Incorrect"}
    )
    fig = px.bar(
        counts,
        x="Count",
        y="Choice",
        orientation="h",
        color="answer_key",
        color_discrete_map={"Correct": COLOR_PASS, "Incorrect": "#1976D2"},
        labels={"answer_key": "Answer key", "Choice": "Answer choice"},
        title=chart_title,
        custom_data=["Answer hover"],
    )
    chart_height = max(320, len(counts) * 36)
    fig.update_layout(
        height=chart_height,
        yaxis={
            "categoryorder": "total ascending",
            "title": "",
            "automargin": True,
        },
        showlegend=True,
    )
    fig.update_traces(
        hovertemplate=(
            "<b>%{y}</b><br>"
            "%{customdata[0]}"
            "<br><br>Count: %{x}"
            "<extra></extra>"
        )
    )
    _apply_answer_key_legend(fig)
    _apply_hover_layout(fig)
    _plotly_chart(fig, use_container_width=True, responsive=False)

    legend = counts.copy()
    legend["Correct"] = legend["is_correct"].map({True: "Yes", False: "No"})
    st.dataframe(
        legend[legend_cols],
        use_container_width=True,
        hide_index=True,
    )

    if looks_like_mcq_multiselect(answer_col):
        combo = answer_col.value_counts().reset_index()
        combo.columns = ["Response combination", "Count"]
        with st.expander("All response combinations (advanced)"):
            st.caption(
                "Each row is exactly what one or more students submitted "
                "(e.g. B only, or B + C + D together)."
            )
            st.dataframe(combo, use_container_width=True, hide_index=True)





def _student_table(df: pd.DataFrame) -> pd.DataFrame:

    display_cols = ["section", "submitted", "score", "n correct", "n incorrect"]

    cols = [c for c in display_cols if c in df.columns]

    out = df[cols].copy()

    out.insert(0, "Anonymous ID", out.index.astype(str))

    if "submitted" in out.columns:

        out["submitted"] = out["submitted"].dt.strftime("%Y-%m-%d %H:%M UTC")

    return out





def _open_ended_student_breakdown(

    df: pd.DataFrame,

    student_id: str,

    open_ended_meta: list[dict[str, Any]],

) -> None:

    """Show only ungraded open-ended answers for one student."""

    if student_id not in df.index:

        return

    row = df.loc[student_id]

    rows: list[dict[str, str]] = []

    for item in open_ended_meta:

        qcol = item["question_col"]

        prompt = truncate_label(item["question"], max_len=80)

        raw = row.get(qcol, pd.NA)

        if pd.isna(raw) or not str(raw).strip():

            answer = "No answer"

        else:

            answer = normalize_open_text(raw)

        rows.append(

            {

                "Prompt": item["label"],

                "Question (short)": prompt,

                "Response": answer,

            }

        )

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)





def _student_breakdown(df: pd.DataFrame, student_id: str, questions_meta: list[dict[str, Any]]) -> None:

    if student_id not in df.index:

        return

    row = df.loc[student_id]

    rows = []

    for q in questions_meta:

        ans = row.get(q["question_col"], pd.NA)

        if pd.isna(ans):

            ans = "No answer"

        score = pd.to_numeric(row.get(q["score_col"], pd.NA), errors="coerce")

        got_it = pd.notna(score) and score >= q["max_score"] - 1e-6

        rows.append(

            {

                "Question": q["q_label"],

                "Answer": str(ans),

                "Score": score,

                "Max": q["max_score"],

                "Correct": "Yes" if got_it else "No",

            }

        )

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)





def _focus_student_from_open_ended(anon_id: str, focus_key: str) -> None:

    """Store the student to highlight in Student Detail and open-ended views."""

    st.session_state[focus_key] = anon_id

    st.session_state["focus_student_id"] = anon_id

    st.session_state["student_detail_pick"] = anon_id





def _response_preview(text: str, max_len: int = 80) -> str:

    """One-line preview for selectbox labels."""

    one_line = str(text).replace("\n", " ").strip()

    if len(one_line) <= max_len:

        return one_line

    return one_line[: max_len - 3] + "..."


def _theme_roster_df(theme_df: pd.DataFrame, preview_len: int = 120) -> pd.DataFrame:
    """Table of every student in a theme with a short response preview."""
    rows = [
        {
            "Student": str(row["Anonymous ID"]),
            "Response preview": _response_preview(str(row["Response"]), max_len=preview_len),
        }
        for _, row in theme_df.iterrows()
    ]
    return pd.DataFrame(rows).sort_values("Student").reset_index(drop=True)





def _wrap_for_hover(text: str, width: int = 52) -> str:

    """Wrap long question text with HTML line breaks for Plotly tooltips."""

    words = str(text).replace("\n", " ").split()

    if not words:

        return ""

    lines: list[str] = []

    current: list[str] = []

    line_len = 0

    for word in words:

        add = len(word) + (1 if current else 0)

        if current and line_len + add > width:

            lines.append(" ".join(current))

            current = [word]

            line_len = len(word)

        else:

            current.append(word)

            line_len += add

    if current:

        lines.append(" ".join(current))

    return "<br>".join(lines)


def _apply_answer_key_legend(fig) -> None:
    """Legend for correct vs incorrect options (Question analysis bar charts)."""
    fig.update_layout(
        legend=dict(
            title=dict(text="Answer key"),
            orientation="v",
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=1.02,
            font=dict(size=11),
            tracegroupgap=4,
        ),
        margin=dict(r=120, l=72, t=55, b=50),
    )


def _apply_hover_layout(fig) -> None:
    """Consistent wide, left-aligned Plotly hover tooltips."""
    fig.update_layout(
        hoverlabel=dict(
            align="left",
            bgcolor="white",
            bordercolor="#cccccc",
            font_size=12,
            namelength=-1,
        ),
    )


def _question_hover_series(texts: pd.Series) -> pd.Series:
    """Wrap question prompts for Plotly ``customdata`` hovers."""
    return texts.map(lambda t: _wrap_for_hover(str(t).replace("\n", " ")))


def _finish_question_bar_hover(fig, value_label: str) -> None:
    """Wrapped full question text on horizontal bar charts (Q1, Q2, ...)."""
    fig.update_traces(
        hovertemplate=(
            f"<b>%{{y}}</b><br>"
            f"%{{customdata[0]}}"
            f"<br><br>{value_label}: %{{x:.1f}}"
            "<extra></extra>"
        )
    )
    _apply_hover_layout(fig)


def _finish_wrapped_label_hover(fig, label_name: str, value_name: str, value_fmt: str) -> None:
    """Wrapped hover for charts with a long categorical label (topic, answer)."""
    fig.update_traces(
        hovertemplate=(
            f"<b>%{{y}}</b><br>"
            f"%{{customdata[0]}}"
            f"<br><br>{value_name}: {value_fmt}"
            "<extra></extra>"
        )
    )
    _apply_hover_layout(fig)


def _finish_simple_hover(fig, template: str) -> None:
    fig.update_traces(hovertemplate=template)
    _apply_hover_layout(fig)



def _render_open_student_content(

    df: pd.DataFrame,

    student_id: str,

    prompt_text: str,

    open_ended_meta: list[dict[str, Any]],

    text_key: str,

) -> None:

    """Show full open-ended text for one student (current prompt + all open prompts)."""

    st.markdown(f"**{student_id}**")

    _render_response_text_area(prompt_text, key=text_key)

    st.caption("All open-ended answers for this student:")

    _open_ended_student_breakdown(df, student_id, open_ended_meta)





def _inject_response_text_styles() -> None:
    """Once-per-session CSS so full-response text is high-contrast (not disabled gray)."""
    if st.session_state.get("_response_text_css"):
        return
    st.markdown(
        """
        <style>
        .quiz-response-text {
            background: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 8px;
            padding: 1rem 1.25rem;
            color: #1a1a1a;
            font-size: 1rem;
            line-height: 1.6;
            white-space: pre-wrap;
            word-wrap: break-word;
            max-height: 480px;
            overflow-y: auto;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.session_state["_response_text_css"] = True


def _render_readable_response(text: str) -> None:
    """Display student response text with readable contrast."""
    _inject_response_text_styles()
    safe = html.escape(str(text).strip())
    st.markdown(f'<div class="quiz-response-text">{safe}</div>', unsafe_allow_html=True)


def _show_full_response(
    text: str,
    student_id: str,
    area_key: str,
    *,
    nested: bool = False,
) -> None:
    """Prominent scrollable panel for a student's complete open-ended answer."""
    del area_key  # layout key not needed for markdown block
    if nested:
        st.markdown(f"**Full response - {student_id}**")
        _render_readable_response(text)
        return
    with st.expander(f"Full response - {student_id}", expanded=True):
        _render_readable_response(text)


def _render_response_text_area(text: str, key: str) -> None:
    """Show a scrollable full response."""
    del key
    _render_readable_response(text)





def _render_open_ended_section(

    df: pd.DataFrame,

    open_ended_meta: list[dict[str, Any]],

) -> None:

    """Open-ended prompts: response stats, word frequencies, themes, samples."""

    st.caption(

        "Ungraded reflection and feedback questions (score column is always 0). "

        "Analysis uses Python built-in text tools only; responses are anonymised."

    )

    if not open_ended_meta:

        st.info(
            "No open-ended columns detected in this CSV "
            "(ungraded reflection/feedback prompts with ColumnNN score 0). "
            + diagnose_quiz_columns(df)
        )

        return



    labels = [m["label"] for m in open_ended_meta]

    choice = st.selectbox("Select open-ended prompt", range(len(labels)), format_func=lambda i: labels[i])

    meta = open_ended_meta[choice]

    col = meta["question_col"]



    st.markdown(f"**{meta['label']}**")

    with st.expander("Full prompt text"):

        st.write(meta["question"])



    if col not in df.columns:

        st.warning("This column is not available in the filtered data.")

        return



    stats = analyze_open_text(
        df[col],
        sample_seed=choice + 42,
        prompt_text=meta["question"],
    )



    m1, m2, m3, m4 = st.columns(4)

    m1.metric("Responses", f"{stats['n_answered']:,} / {stats['n_total']:,}")

    m2.metric("Response rate", f"{stats['response_rate_pct']:.1f}%")

    m3.metric("Median length (chars)", f"{stats['median_chars']:.0f}")

    m4.metric("Median length (words)", f"{stats['median_words']:.0f}")



    chart_l, chart_r = st.columns(2)

    with chart_l:

        if stats["top_words"]:

            wdf = pd.DataFrame(stats["top_words"], columns=["Word", "Count"])

            fig = px.bar(

                wdf.head(15),

                x="Count",

                y="Word",

                orientation="h",

                title="Most frequent words (stopwords removed)",

                color="Count",

                color_continuous_scale="Blues",

                labels={"Word": "", "Count": "Count"},

            )

            fig.update_layout(
                yaxis={
                    "title": None,
                    "categoryorder": "total ascending",
                    "automargin": True,
                },
                xaxis={"title": {"text": "Count", "standoff": 10}, "automargin": True},
                margin=dict(l=120, r=80, t=55, b=55),
                coloraxis_colorbar=dict(title=None, thickness=14, len=0.55),
            )
            _finish_simple_hover(fig, "Word: %{y}<br>Count: %{x}<extra></extra>")

            _plotly_chart(fig, use_container_width=True)

        else:

            st.info("Not enough text for word frequency analysis.")



    with chart_r:

        if stats["word_lengths"]:

            ldf = pd.DataFrame({"Words per response": stats["word_lengths"]})

            fig = px.histogram(

                ldf,

                x="Words per response",

                nbins=20,

                title="Response length distribution (words)",

                color_discrete_sequence=px.colors.sequential.Blues[4:5],

            )
            _finish_simple_hover(fig, "Words: %{x}<br>Students: %{y}<extra></extra>")

            _plotly_chart(fig, use_container_width=True)

        else:

            st.info("No responses to plot length distribution.")



    responses_df = collect_open_responses(df, col)



    if stats["theme_counts"]:

        st.subheader("Topic mentions")

        theme_kinds = stats.get("theme_kinds") or {}
        chart_counts = {
            k: v
            for k, v in stats["theme_counts"].items()
            if theme_kinds.get(k, "curated") == "curated"
        }
        if not chart_counts:
            chart_counts = stats["theme_counts"]

        tdf = pd.DataFrame(
            [
                {"Topic": k, "Responses": v}
                for k, v in sort_theme_counts_for_display(chart_counts, theme_kinds)
            ]
        )
        tdf["Topic hover"] = tdf["Topic"].map(lambda t: _wrap_for_hover(str(t), width=48))

        n_topics = len(tdf)

        chart_height = max(420, n_topics * 32)

        max_label_len = max((len(str(t)) for t in tdf["Topic"]), default=20)

        left_margin = min(460, max(180, max_label_len * 7 + 24))

        fig = px.bar(

            tdf,

            x="Responses",

            y="Topic",

            orientation="h",

            color="Responses",

            color_continuous_scale="Blues",

            title="Topics mentioned in open-ended responses",

            custom_data=["Topic hover"],

            labels={"Topic": "", "Responses": "Responses"},

        )

        fig.update_layout(

            height=chart_height,

            yaxis={

                "title": None,

                "categoryorder": "total ascending",

                "automargin": True,

                "tickfont": {"size": 12},

            },

            xaxis={

                "title": {"text": "Responses", "standoff": 10},

                "automargin": True,

            },

            margin=dict(l=left_margin, r=80, t=55, b=55),

            coloraxis_colorbar=dict(

                title=None,

                thickness=14,

                len=0.55,

            ),

        )
        _finish_wrapped_label_hover(fig, "Topic", "Responses", "%{x}")

        _plotly_chart(fig, use_container_width=True)

        with st.expander("Full topic list (all titles)"):

            st.dataframe(

                tdf.sort_values("Responses", ascending=False),

                use_container_width=True,

                hide_index=True,

            )



    focus_key = f"open_focus_student_{choice}"

    valid_ids = responses_df["Anonymous ID"].astype(str).tolist()

    focused = st.session_state.get(focus_key)



    if focused and focused not in valid_ids:

        st.session_state.pop(focus_key, None)

        focused = None



    if not responses_df.empty:

        grouped, _theme_kind = group_responses_by_theme(
            responses_df,
            prompt_text=meta["question"],
        )

        theme_order = sorted(
            grouped.keys(),
            key=lambda label: (-len(grouped[label]), label.lower()),
        )



        st.subheader("Responses by theme")

        st.caption(
            "Curated library topics only (e.g. APA, ProQuest, Copyright). Generic words from "
            "reflection phrasing (e.g. taught, careful, various) are not listed as themes. "
            "Listed from most to fewest responses. "
            "Click once on a table row to open the full response below (double-click is not needed)."
        )

        for theme_label in theme_order:

            theme_df = grouped[theme_label]

            theme_slug = re.sub(r"[^\w]+", "_", theme_label)

            theme_ids = theme_df["Anonymous ID"].astype(str).tolist()

            expand = bool(focused and focused in theme_ids)

            with st.expander(

                f"{theme_label} ({len(theme_df)} responses)",

                expanded=expand,

            ):

                if len(theme_label) > 50:

                    st.caption(theme_label)

                roster = _theme_roster_df(theme_df)
                response_by_student = {
                    str(r["Anonymous ID"]): str(r["Response"])
                    for _, r in theme_df.iterrows()
                }

                st.markdown(f"**{len(roster)} students** in this theme")

                table_key = f"open_theme_table_{choice}_{theme_slug}"
                pick_key = f"open_theme_pick_{choice}_{theme_slug}"

                table_event = st.dataframe(
                    roster,
                    use_container_width=True,
                    hide_index=True,
                    on_select="rerun",
                    selection_mode="single-row",
                    key=table_key,
                )

                if table_event.selection.rows:
                    st.session_state[pick_key] = str(
                        roster.iloc[table_event.selection.rows[0]]["Student"]
                    )

                selected_id = st.selectbox(
                    "Student",
                    theme_ids,
                    format_func=lambda aid: str(aid),
                    key=pick_key,
                )
                selected_id = str(selected_id)

                if selected_id in response_by_student:
                    safe_id = re.sub(r"[^\w]+", "_", selected_id)
                    _show_full_response(
                        response_by_student[selected_id],
                        selected_id,
                        area_key=f"open_theme_full_{choice}_{theme_slug}_{safe_id}",
                        nested=True,
                    )
                    if st.button(
                        "Link to Student Detail tab",
                        key=f"open_theme_link_{choice}_{theme_slug}_{safe_id}",
                    ):
                        _focus_student_from_open_ended(selected_id, focus_key)
                        st.success(f"{selected_id} is now selected in **Student Detail**.")
                else:
                    st.info("Click a row in the table above to read the full response.")



        st.subheader("Browse all responses")

        st.caption(
            "Pick a student below, or use Responses by theme above. "
            "The full answer opens in the panel underneath."
        )



        ids = responses_df["Anonymous ID"].astype(str).tolist()

        browse_key = f"open_browse_{choice}"

        if focused and focused in ids:

            st.session_state[browse_key] = focused



        def _response_label(anon_id: str) -> str:

            text = responses_df.loc[

                responses_df["Anonymous ID"].astype(str) == str(anon_id), "Response"

            ].iloc[0]

            return f"{anon_id} - {_response_preview(text)}"



        picked = st.selectbox(

            "Select student",

            ids,

            format_func=_response_label,

            key=browse_key,

        )

        picked_text = responses_df.loc[

            responses_df["Anonymous ID"].astype(str) == str(picked), "Response"

        ].iloc[0]

        safe_picked = re.sub(r"[^\w]+", "_", str(picked))

        _show_full_response(
            str(picked_text),
            str(picked),
            area_key=f"open_browse_full_{choice}_{safe_picked}",
        )

        st.caption("All open-ended answers for this student:")
        _open_ended_student_breakdown(df, str(picked), open_ended_meta)

        if st.button(

            "Link to Student Detail tab",

            key=f"open_browse_link_{choice}_{safe_picked}",

            type="primary",

        ):

            _focus_student_from_open_ended(str(picked), focus_key)

            st.success(f"{picked} is now selected in **Student Detail**.")





def _section_comparison(df: pd.DataFrame, questions_meta: list[dict[str, Any]]) -> None:

    if "section" not in df.columns or not questions_meta:

        return

    if df.empty:

        st.info("No students match the current filters.")

        return

    long_rows = []

    for q in questions_meta:

        for section, grp in df.groupby("section"):

            scores = pd.to_numeric(grp[q["score_col"]], errors="coerce")

            question_text = str(q["question"]).replace("\n", " ").strip()

            long_rows.append(

                {

                    "Section": section,

                    "Question": q["q_label"],

                    "Question text": question_text,

                    "Question hover": _wrap_for_hover(question_text),

                    "Mean score": scores.mean(),

                    "Max": q["max_score"],

                }

            )

    long_df = pd.DataFrame(long_rows)

    long_df["Mean %"] = (long_df["Mean score"] / long_df["Max"]) * 100

    sections_ordered = sorted(long_df["Section"].astype(str).unique())

    fig = px.bar(

        long_df,

        x="Question",

        y="Mean %",

        color="Section",

        barmode="group",

        custom_data=["Question hover"],

        labels={"Mean %": "Mean score (% of max)"},

        title="Mean score per question by section",

        color_discrete_map=_section_color_map(sections_ordered),

        category_orders={"Section": sections_ordered},

    )

    fig.update_traces(

        hovertemplate=(

            "<b>%{x}</b><br>"

            "%{customdata[0]}"

            "<br><br>Section: %{fullData.name}"

            "<br>Mean: %{y:.1f}%% of max"

            "<extra></extra>"

        )

    )

    n_sec = len(sections_ordered)
    _apply_section_legend(fig, n_sec)
    _apply_hover_layout(fig)

    st.caption(
        "Sections are labeled T01-T28 (tutorial groups), not T0. "
        "Filter sections in the sidebar to reduce legend size. "
        "Legend entries are labels only (not clickable)."
    )
    _plotly_chart(fig, use_container_width=True, responsive=n_sec <= 12)



    q_ref = pd.DataFrame(

        [

            {"Question": q["q_label"], "Full question text": str(q["question"]).replace("\n", " ").strip()}

            for q in questions_meta

        ]

    )

    with st.expander("Question reference (full text for Q1-Q20)"):

        st.dataframe(q_ref, use_container_width=True, hide_index=True)



    if "score" in df.columns:
        summary_rows = []
        for section, grp in df.groupby("section"):
            scores = pd.to_numeric(grp["score"], errors="coerce")
            summary_rows.append(
                {
                    "Section": section,
                    "N students": len(grp),
                    "Mean score": scores.mean(),
                    "Median": scores.median(),
                    "Pass rate %": (scores >= PASS_THRESHOLD).mean() * 100,
                    "Std dev": scores.std(),
                }
            )
        st.subheader("Section summary")
        st.dataframe(
            pd.DataFrame(summary_rows).round(2),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.caption(
            "Section summary by overall score is unavailable for this export "
            "(no score column and no scored questions detected)."
        )





def main() -> None:

    st.title("LANG Quiz Results Dashboard")

    st.caption(

        "Upload a Canvas quiz CSV to explore scores - student identifiers are never stored or shown."

    )



    with st.sidebar:

        st.header("Data")

        uploaded = st.file_uploader("Upload quiz CSV", type=["csv"])

        st.info(

            "\U0001f512 **Privacy:** Uploaded data is processed in memory only. "

            "Student names, IDs, and other identifiers are removed immediately on upload "

            "and are never stored or displayed."

        )



    with st.expander("\u2139\ufe0f How to use this dashboard", expanded=False):

        st.markdown(

            """

1. Export your Canvas quiz results as CSV (Gradebook \u2192 Quiz Statistics \u2192 Download)

2. Upload the CSV using the sidebar uploader

   \u2192 Student names, IDs, emails, and other identifiers are automatically removed the moment the file is read.

   \u2192 The original CSV file on your computer is not modified.

3. Use the filters to drill into specific sections, score ranges, or dates

4. Explore tabs for overview, question-level, open-ended feedback, student-level, and section comparison views

5. **Data Privacy:** Do not keep raw Canvas CSV exports containing student PII longer than necessary.

   Per institutional policy, clear out names, IDs, emails and barcodes from any saved files as soon as possible,

   and delete Zoom attendance records with personal data within one year of the session.

            """

        )



    if uploaded is None:

        st.markdown(

            """

### Welcome



Use the **sidebar** to upload a Canvas quiz results CSV.



- Encoding and question columns are detected automatically.

- Personally identifying information is stripped on upload.

- No data is written to disk.

            """

        )

        return



    try:

        with st.spinner("Loading and scrubbing data..."):

            file_bytes = bytes(uploaded.getvalue())
            file_hash = hashlib.sha256(file_bytes).hexdigest()

            df, questions_meta, open_ended_meta, sections, _pii_report = cached_load(
                file_hash, file_bytes
            )

            df = shuffle_anonymous(df)

            if st.session_state.get("_quiz_file_hash") != file_hash:
                st.session_state["_quiz_file_hash"] = file_hash
                _init_section_filter_state(sections, select_all=True)

    except Exception as exc:

        st.error(
            "Could not load this file. Upload the original Canvas quiz CSV export "
            "(do not open and re-save in Excel first). "
            f"Details: {exc}"
        )

        return



    defaults = _default_filter_state(sections, df)



    with st.sidebar:

        st.header("Filters")

        if st.button("Reset filters"):

            _init_section_filter_state(sections, select_all=True)

            for key in ("score_range", "date_range", "pass_filter"):

                st.session_state.pop(key, None)

            st.rerun()



        sel_sections = _section_filter_sidebar(sections)

        score_range = st.slider(

            "Score range (%)",

            min_value=0.0,

            max_value=100.0,

            value=defaults["score_range"],

            key="score_range",

        )

        dmin, dmax = defaults["date_range"]

        date_range = st.date_input(

            "Submission dates",

            value=(dmin, dmax),

            min_value=dmin,

            max_value=dmax,

            key="date_range",

        )

        pass_filter = st.radio(

            "Show only",

            options=["All students", "Passed (\u226560)", "Failed (<60)"],

            index=0,

            key="pass_filter",

        )



    if isinstance(date_range, tuple) and len(date_range) == 2:

        start = pd.Timestamp(date_range[0], tz="UTC")

        end = pd.Timestamp(date_range[1], tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

        date_tuple = (start, end)

    else:

        date_tuple = (

            pd.Timestamp(defaults["date_range"][0], tz="UTC"),

            pd.Timestamp(defaults["date_range"][1], tz="UTC") + pd.Timedelta(days=1),

        )



    active_filters = {

        "sections": sel_sections,

        "score_range": score_range,

        "date_range": (date_range[0], date_range[1]) if isinstance(date_range, tuple) else defaults["date_range"],

        "pass_filter": pass_filter,

    }

    filtered = apply_filters(df, sel_sections, score_range, date_tuple, pass_filter)

    _filter_banner(active_filters, defaults)

    if not sel_sections:

        st.warning("No sections selected. Select at least one section to view data.")

        st.stop()



    tab_overview, tab_questions, tab_open, tab_students, tab_sections = st.tabs(

        [

            "\U0001f4ca Overview",

            "\u2753 Question Analysis",

            "\U0001f4dd Open-ended",

            "\U0001f464 Student Detail",

            "\U0001f4cb Section Comparison",

        ]

    )



    with tab_overview:

        _kpi_row(filtered)

        col_a, col_b = st.columns(2)

        with col_a:

            _score_histogram(filtered)

        with col_b:

            _score_by_section_box(filtered)

        _submission_timeline(filtered)



    with tab_questions:

        st.caption("Scored multiple-choice questions only. See the **Open-ended** tab for reflection and feedback text.")

        col1, col2 = st.columns(2)

        with col1:

            _question_full_marks_chart(filtered, questions_meta)

        with col2:

            _question_avg_score_chart(filtered, questions_meta)

        st.divider()

        _question_detail(filtered, questions_meta)



    with tab_open:

        _render_open_ended_section(filtered, open_ended_meta)



    with tab_students:

        st.caption(

            "Student identities are not stored or displayed. "

            "Rows are randomly ordered and labelled anonymously."

        )

        focus_id = st.session_state.get("focus_student_id")

        if focus_id:

            st.success(

                f"Showing **{focus_id}** (selected from Open-ended). "

                "Change the dropdown below to view another student."

            )

        table_df = _student_table(filtered)

        st.dataframe(table_df, use_container_width=True, hide_index=True)

        student_options = table_df["Anonymous ID"].tolist()

        if student_options:

            if focus_id and focus_id in student_options:

                st.session_state["student_detail_pick"] = focus_id

            elif "student_detail_pick" not in st.session_state:

                st.session_state["student_detail_pick"] = student_options[0]

            selected = st.selectbox(

                "View answer breakdown for",

                student_options,

                key="student_detail_pick",

            )

            _student_breakdown(filtered, selected, questions_meta)



    with tab_sections:

        _section_comparison(filtered, questions_meta)

        st.download_button(

            label="Download filtered data (CSV, PII-free)",

            data=export_scrubbed_csv(filtered),

            file_name="quiz_results_anonymised.csv",

            mime="text/csv",

        )





if __name__ == "__main__":

    main()

