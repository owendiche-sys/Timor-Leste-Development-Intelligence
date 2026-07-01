"""Accessible Streamlit dashboard for the saved development-intelligence artifacts."""

from __future__ import annotations

import html
import hashlib
import json
import re
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# ---------------------------------------------------------------------
# Paths and colour palette
# ---------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "outputs" / "results"
PROCESSED = ROOT / "outputs" / "processed"
FIGURES = ROOT / "outputs" / "figures"

COLORS = {
    "navy": "#17324D",
    "blue": "#176B87",
    "teal": "#1C887A",
    "gold": "#C98210",
    "coral": "#C4513A",
    "surface": "#F5F7F8",
    "text": "#14212B",
    "muted": "#5B6770",
    "grid": "#DCE3E7",
}

THEME_COLORS = {
    "education": "#176B87",
    "health": "#1C887A",
    "inequality": "#7A5195",
    "economy": "#C98210",
    "gender": "#C4513A",
    "climate": "#4F772D",
    "employment": "#3D5A80",
}

SERIES_LABEL_OVERRIDES = {
    "2. 1 + education": "Projected population — policy scenario 2 (economy + education)",
    "Health, % ($, millions)": "Health budget share, %",
    "Agriculture, % ($, millions)": "Agriculture budget share, %",
}

TREND_COLORS = [
    "#176B87",
    "#C4513A",
    "#1C887A",
    "#C98210",
    "#7A5195",
    "#3D5A80",
    "#4F772D",
]


# ---------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------

def read_json(path: Path, default):
    """Read JSON safely. Return default if the file is missing or invalid."""
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default


def read_jsonl(path: Path) -> list[dict]:
    """Read JSONL safely. Return an empty list if the file is missing."""
    if not path.exists():
        return []

    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def clean_text(value, fallback: str = "") -> str:
    """Return a safe stripped string for display."""
    if pd.isna(value):
        return fallback
    value = str(value).strip()
    return value if value else fallback


def readable_name(value) -> str:
    """Convert code-style labels into dashboard-friendly labels."""
    return clean_text(value).replace("_", " ").strip().title()


def readable_series_name(value) -> str:
    """Clarify terse table-row labels without changing the extracted source data."""
    value = clean_text(value, "Unnamed series")
    return SERIES_LABEL_OVERRIDES.get(value, value)


def trend_family(value) -> str:
    """Group trend series that are meaningful to compare together."""
    name = readable_series_name(value).lower()
    if "projected population" in name or "policy scenario" in name:
        return "Population projections"
    if "state budget total" in name or name == "state budget, $, millions":
        return "Budget totals"
    if "literacy" in name or "years of schooling" in name:
        return "Education outcomes"
    return "Budget shares"


def year_position(value) -> float | None:
    """Convert report year and fiscal-period labels to a chronological position."""
    label = clean_text(value)
    short_range = re.fullmatch(r"((?:19|20)\d{2})/(\d{2})", label)
    if short_range:
        start = int(short_range.group(1))
        end = (start // 100) * 100 + int(short_range.group(2))
        return (start + end) / 2
    years = [int(year) for year in re.findall(r"(?:19|20)\d{2}", label)]
    return sum(years) / len(years) if years else None


def section_title_for_page(value) -> str:
    """Return the report section containing a PDF page."""
    try:
        page = int(value)
    except (TypeError, ValueError):
        return "Section not identified"
    for section in read_json(PROCESSED / "sections.json", []):
        if section.get("start_pdf_page", 0) <= page <= section.get("end_pdf_page", 0):
            return clean_text(section.get("title"), readable_name(section.get("section_id")))
    return "Section not identified"


def shorten_label(value, max_length: int = 55) -> str:
    """Shorten very long labels for chart legends and axes."""
    value = clean_text(value)
    if len(value) <= max_length:
        return value
    return value[:max_length].rstrip() + "..."


def format_number(value) -> str:
    """Format numbers cleanly for metric cards and tables."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return clean_text(value, "-")

    if abs(value) >= 1000:
        return f"{value:,.0f}"
    return f"{value:.2f}".rstrip("0").rstrip(".")


# ---------------------------------------------------------------------
# Plot styling
# ---------------------------------------------------------------------

def chart_layout(
    figure: go.Figure,
    title: str,
    height: int = 520,
    x_title: str | None = None,
    y_title: str | None = None,
    show_legend: bool = True,
    left_margin: int = 110,
    right_margin: int = 60,
    top_margin: int = 95,
    bottom_margin: int = 95,
) -> go.Figure:
    """Apply one consistent readable layout to all Plotly charts."""

    figure.update_layout(
        title={
            "text": title,
            "font": {"size": 24, "color": COLORS["navy"]},
            "x": 0.01,
            "xanchor": "left",
        },
        height=height,
        margin={
            "l": left_margin,
            "r": right_margin,
            "t": top_margin,
            "b": bottom_margin,
        },
        paper_bgcolor="white",
        plot_bgcolor="white",
        font={
            "family": "Arial, sans-serif",
            "color": COLORS["text"],
            "size": 16,
        },
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.04,
            "xanchor": "center",
            "x": 0.5,
            "font": {"size": 14, "color": COLORS["text"]},
        },
        hoverlabel={"font_size": 14},
        showlegend=show_legend,
    )

    figure.update_xaxes(
        title={
            "text": x_title or "",
            "font": {"size": 17, "color": COLORS["navy"]},
            "standoff": 18,
        },
        tickfont={"size": 14, "color": COLORS["text"]},
        showgrid=True,
        gridcolor=COLORS["grid"],
        linecolor=COLORS["text"],
        tickcolor=COLORS["text"],
        zeroline=False,
        automargin=True,
    )

    figure.update_yaxes(
        title={
            "text": y_title or "",
            "font": {"size": 16, "color": COLORS["navy"]},
            "standoff": 28,
        },
        tickfont={"size": 14, "color": COLORS["text"]},
        showgrid=False,
        linecolor=COLORS["text"],
        tickcolor=COLORS["text"],
        zeroline=False,
        automargin=True,
    )

    return figure


def show_plot(figure: go.Figure) -> None:
    """Save a high-resolution PNG and render a Plotly chart in Streamlit."""
    FIGURES.mkdir(parents=True, exist_ok=True)

    title = figure.layout.title.text or "dashboard-chart"
    title = re.sub(r"<[^>]+>", "", str(title))
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-") or "dashboard-chart"
    content_hash = hashlib.sha256(figure.to_json().encode("utf-8")).hexdigest()[:8]
    try:
        figure.write_image(
            FIGURES / f"{slug}-{content_hash}.png",
            width=1600,
            height=int(figure.layout.height or 900),
            scale=2,
        )
    except RuntimeError:
        # Kaleido requires Chrome, which may be unavailable on hosted deployments.
        # Figures can still be generated locally and committed to the repository.
        pass

    st.plotly_chart(
        figure,
        width="stretch",
        config={"displaylogo": False},
    )


# ---------------------------------------------------------------------
# Data preparation functions
# ---------------------------------------------------------------------

def theme_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create theme totals and theme map dataframes."""

    totals = read_json(RESULTS / "theme_llm_totals.json", [])
    extractor_model = read_json(RESULTS / "run_manifest.json", {}).get(
        "extractor_model", "not recorded"
    )
    method = "Qwen2.5 semantic classification"

    if not totals:
        totals = read_json(RESULTS / "theme_keyword_totals.json", [])
        method = "Keyword baseline"

    total_frame = pd.DataFrame(totals)

    if not total_frame.empty:
        if "count" not in total_frame.columns:
            total_frame["count"] = 0
        if "theme" not in total_frame.columns:
            total_frame["theme"] = "unknown"
        total_frame["method"] = method
        total_frame["count"] = pd.to_numeric(total_frame["count"], errors="coerce").fillna(0)
        total_frame["theme"] = total_frame["theme"].astype(str).str.lower()

    classifications = read_jsonl(RESULTS / "theme_classification.jsonl")
    long_rows = []

    for row in classifications:
        for theme, detail in row.get("themes", {}).items():
            if int(detail.get("present", 0)) == 1:
                long_rows.append(
                    {
                        "theme": str(theme).lower(),
                        "pdf_page": detail.get("pdf_page") or row.get("start_pdf_page"),
                        "section": readable_name(row.get("section_id", "unknown")),
                        "evidence": detail.get("evidence", ""),
                        "extractor_model": extractor_model,
                    }
                )

    if not long_rows:
        baseline = read_jsonl(RESULTS / "theme_keyword_baseline.jsonl")
        long_rows = [
            {
                "theme": str(row.get("theme", "unknown")).lower(),
                "pdf_page": row.get("start_pdf_page"),
                "section": readable_name(row.get("section_id", "unknown")),
                "evidence": ", ".join(row.get("matched_keywords", [])),
                "extractor_model": "deterministic keyword baseline",
            }
            for row in baseline
            if int(row.get("present", 0)) == 1
        ]

    map_frame = pd.DataFrame(long_rows)

    if not map_frame.empty:
        map_frame["pdf_page"] = pd.to_numeric(map_frame["pdf_page"], errors="coerce")
        map_frame = map_frame.dropna(subset=["pdf_page"])
        map_frame["pdf_page"] = map_frame["pdf_page"].astype(int)

    return total_frame, map_frame


def indicators_frame() -> pd.DataFrame:
    """Create dataframe of extracted numerical indicators."""

    rows = read_json(RESULTS / "indicators.json", {}).get("indicators", [])
    frame = pd.DataFrame(rows)

    if frame.empty:
        return frame

    for column in ["name", "value", "unit", "year", "population_group", "pdf_page", "evidence"]:
        if column not in frame.columns:
            frame[column] = ""

    frame["numeric_value"] = pd.to_numeric(frame["value"], errors="coerce")

    frame["population_group"] = frame["population_group"].fillna("").astype(str)
    frame["year"] = frame["year"].fillna("").astype(str)
    frame["source_section"] = frame["pdf_page"].apply(section_title_for_page)

    frame["label"] = (
        frame["year"].where(frame["year"].str.strip() != "", "No year")
        + " | "
        + frame["population_group"].where(
            frame["population_group"].str.strip() != "",
            "All population",
        )
    )

    return frame


def time_series_frame() -> pd.DataFrame:
    """Create dataframe of extracted time-series values."""

    payload = read_json(RESULTS / "time_series.json", {})

    if not payload:
        payload = read_json(RESULTS / "deterministic_time_series.json", {})

    rows = []

    for series in payload.get("series", []):
        raw_series_name = series.get("series_name", "Unnamed series")
        for point in series.get("points", []):
            rows.append(
                {
                    "series": readable_series_name(raw_series_name),
                    "source_series_label": raw_series_name,
                    "year": point.get("year", ""),
                    "value": point.get("value"),
                    "unit": series.get("unit", "as reported"),
                    "pdf_page": series.get("pdf_page"),
                }
            )

    frame = pd.DataFrame(rows)

    if frame.empty:
        return frame

    frame["year_label"] = frame["year"].astype(str)
    frame["year_numeric"] = pd.to_numeric(frame["year"], errors="coerce")
    frame["year_position"] = frame["year_label"].apply(year_position)
    frame["numeric_value"] = pd.to_numeric(frame["value"], errors="coerce")
    frame = frame.dropna(subset=["numeric_value"])
    frame["trend_family"] = frame["series"].apply(trend_family)
    frame = frame.sort_values(["series", "year_position", "year_label"])

    def index_series(values: pd.Series) -> pd.Series:
        """Index each series so first available value equals 100."""
        first_valid = values.dropna()
        if first_valid.empty:
            return values
        base = first_valid.iloc[0]
        if base == 0:
            return values
        return (values / base) * 100

    frame["indexed_value"] = frame.groupby("series")["numeric_value"].transform(index_series)

    return frame


def evaluation_frame() -> pd.DataFrame:
    """Create dataframe of evaluator scores."""

    rows = []

    for record in read_json(RESULTS / "evaluations.json", []):
        for criterion, score in record.get("scores", {}).items():
            rows.append(
                {
                    "output": readable_name(record.get("output_id", record.get("task", "output"))),
                    "criterion": readable_name(criterion),
                    "score": score,
                    "verdict": record.get("verdict", ""),
                    "model_verdict": record.get("model_verdict", ""),
                    "verdict_policy_applied": record.get("verdict_policy_applied", False),
                    "justification": record.get("justification", ""),
                    "unsupported_claims": "; ".join(record.get("unsupported_claims", [])),
                    "missing_items": "; ".join(record.get("missing_items", [])),
                    "policy_reason": record.get("policy_reason", ""),
                }
            )

    frame = pd.DataFrame(rows)

    if not frame.empty:
        frame["score"] = pd.to_numeric(frame["score"], errors="coerce")

    return frame


# ---------------------------------------------------------------------
# Streamlit page setup and CSS
# ---------------------------------------------------------------------

st.set_page_config(
    page_title="Timor-Leste Development Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
        --navy: #17324D;
        --blue: #176B87;
        --teal: #1C887A;
        --gold: #C98210;
        --coral: #C4513A;
        --page: #FBFCFC;
        --text: #14212B;
        --muted: #5B6770;
        --border: #D8E1E8;
        --focus: #F0A830;
        --shadow: 0 3px 12px rgba(15, 23, 42, 0.05);
    }

    html, body, [class*="css"] {
        font-family: "Inter", "Segoe UI", Arial, sans-serif;
    }

    .stApp {
        background: var(--page);
        color: var(--text);
    }

    div[data-testid="stAppViewBlockContainer"],
    .block-container {
        max-width: 1180px;
        padding-top: 2.95rem;
        padding-bottom: 3rem;
    }

    header[data-testid="stHeader"] {
        background: rgba(251, 252, 252, 0.94);
    }

    /* Keep the sidebar reopen control visible against the light app header. */
    [data-testid="stExpandSidebarButton"],
    [data-testid="stExpandSidebarButton"] button,
    [data-testid="stSidebarCollapsedControl"],
    [data-testid="stSidebarCollapsedControl"] button {
        color: var(--navy) !important;
        background: #FFFFFF !important;
        border: 1px solid var(--border) !important;
        border-radius: 10px !important;
        box-shadow: var(--shadow);
    }

    [data-testid="stExpandSidebarButton"] svg,
    [data-testid="stExpandSidebarButton"] svg *,
    [data-testid="stExpandSidebarButton"] span,
    [data-testid="stExpandSidebarButton"] i,
    [data-testid="stSidebarCollapsedControl"] svg,
    [data-testid="stSidebarCollapsedControl"] svg *,
    [data-testid="stSidebarCollapsedControl"] span,
    [data-testid="stSidebarCollapsedControl"] i {
        color: var(--navy) !important;
        fill: var(--navy) !important;
        stroke: var(--navy) !important;
        -webkit-text-fill-color: var(--navy) !important;
        opacity: 1 !important;
    }

    [data-testid="stExpandSidebarButton"]:hover,
    [data-testid="stExpandSidebarButton"] button:hover,
    [data-testid="stSidebarCollapsedControl"]:hover,
    [data-testid="stSidebarCollapsedControl"] button:hover {
        background: #EAF3F7 !important;
        border-color: var(--blue) !important;
    }

    [data-testid="stExpandSidebarButton"]:focus-visible,
    [data-testid="stExpandSidebarButton"] button:focus-visible,
    [data-testid="stSidebarCollapsedControl"]:focus-visible,
    [data-testid="stSidebarCollapsedControl"] button:focus-visible {
        outline: 3px solid var(--focus) !important;
        outline-offset: 2px;
    }

    footer,
    div[data-testid="stDecoration"] {
        display: none;
    }

    h1, h2, h3, h4, h5, h6 {
        color: var(--navy);
        letter-spacing: -0.02em;
    }

    p, li, label {
        color: var(--text);
    }

    a:focus-visible,
    button:focus-visible,
    input:focus-visible,
    [role="radio"]:focus-visible,
    [tabindex]:focus-visible {
        outline: 3px solid var(--focus) !important;
        outline-offset: 2px;
    }

    section[data-testid="stSidebar"] {
        background: var(--navy);
    }

    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] [role="radio"] {
        color: #F8FBFF !important;
    }

    section[data-testid="stSidebar"] .stCaptionContainer,
    section[data-testid="stSidebar"] .stCaptionContainer * {
        color: #D7E3EF !important;
    }

    section[data-testid="stSidebar"] hr {
        border-color: rgba(255, 255, 255, 0.2);
    }

    section[data-testid="stSidebar"] [role="radio"] {
        min-height: 44px;
    }

    .hero {
        padding: 1.9rem 2.2rem;
        background: linear-gradient(120deg, var(--navy), var(--blue));
        border-radius: 18px;
        margin: 0 0 1.4rem;
    }

    .hero h1 {
        color: #FFFFFF !important;
        margin: 0;
        font-size: clamp(1.9rem, 4vw, 2.45rem);
        line-height: 1.15;
        font-weight: 800;
    }

    .hero p {
        max-width: 72ch;
        margin: 0.75rem 0 0;
        color: #E9F3F5 !important;
        font-size: 1.08rem;
        line-height: 1.55;
    }

    div[data-testid="stMetric"] {
        min-height: 115px;
        padding: 1.2rem 1.35rem;
        background: #FFFFFF;
        border: 1px solid var(--border);
        border-radius: 16px;
        box-shadow: var(--shadow);
    }

    div[data-testid="stMetricLabel"],
    div[data-testid="stMetricLabel"] * {
        color: var(--muted) !important;
        font-size: 0.95rem !important;
        font-weight: 700 !important;
    }

    div[data-testid="stMetricValue"],
    div[data-testid="stMetricValue"] * {
        color: var(--navy) !important;
        font-size: 2.15rem !important;
        font-weight: 800 !important;
        font-variant-numeric: tabular-nums;
    }

    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: #FFFFFF;
        border-color: var(--border) !important;
        border-radius: 16px;
        box-shadow: var(--shadow);
    }

    .finding-number {
        display: flex;
        width: 38px;
        height: 38px;
        align-items: center;
        justify-content: center;
        margin-top: 0.1rem;
        border-radius: 50%;
        background: var(--navy);
        color: #FFFFFF !important;
        font-weight: 800;
    }

    .finding-label {
        padding-top: 0.55rem;
        color: var(--muted) !important;
        font-size: 0.9rem;
        font-weight: 800;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }

    .finding-text {
        margin: 0.8rem 0 0.9rem;
        color: var(--text) !important;
        font-size: 1.02rem;
        font-weight: 500;
        line-height: 1.65;
    }

    .source-line {
        margin-top: 0.8rem;
        color: var(--muted) !important;
        font-size: 0.9rem;
        font-weight: 700;
    }

    .source-chip {
        display: inline-block;
        margin: 0.35rem 0.3rem 0 0;
        padding: 0.22rem 0.6rem;
        border: 1px solid var(--border);
        border-radius: 999px;
        background: #EAF3F7;
        color: var(--navy) !important;
        font-size: 0.83rem;
        font-weight: 800;
    }

    .next-card-title {
        margin-bottom: 0.4rem;
        color: var(--navy) !important;
        font-size: 1.05rem;
        font-weight: 800;
    }

    .next-card-text {
        color: var(--text) !important;
        font-size: 0.98rem;
        line-height: 1.55;
    }

    .plot-explainer {
        margin: 0.35rem 0 1.15rem;
        padding: 0.95rem 1.1rem;
        background: linear-gradient(135deg, #F7FBFD, #EEF5F8);
        border: 1px solid var(--border);
        border-left: 5px solid var(--blue);
        border-radius: 14px;
        box-shadow: var(--shadow);
    }

    .plot-explainer-title {
        margin-bottom: 0.35rem;
        color: var(--navy) !important;
        font-size: 0.9rem;
        font-weight: 800;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }

    .plot-explainer-text {
        margin: 0;
        color: var(--text) !important;
        font-size: 1rem;
        line-height: 1.65;
    }

    .indicator-card {
        margin: 1rem 0;
        padding: 1.4rem 1.6rem;
        background: #FFFFFF;
        border: 1px solid var(--border);
        border-radius: 16px;
        box-shadow: var(--shadow);
    }

    .indicator-card-label {
        margin-bottom: 0.35rem;
        color: var(--muted) !important;
        font-size: 0.95rem;
        font-weight: 600;
    }

    .indicator-card-value {
        margin-bottom: 0.4rem;
        color: var(--navy) !important;
        font-size: clamp(2rem, 6vw, 2.6rem);
        font-weight: 800;
        line-height: 1.1;
        font-variant-numeric: tabular-nums;
    }

    .indicator-card-meta {
        margin-top: 0.7rem;
        color: var(--text) !important;
        font-size: 1rem;
        line-height: 1.6;
    }

    .status-note {
        padding: 0.75rem 1rem;
        background: #FFF8E8;
        border-left: 4px solid var(--gold);
        border-radius: 6px;
        color: var(--text);
    }

    div[data-testid="stSelectbox"] label,
    div[data-testid="stMultiSelect"] label,
    div[data-testid="stRadio"] label {
        color: var(--text) !important;
        font-weight: 700;
    }

    div[data-baseweb="select"] > div {
        min-height: 44px;
        background: #FFFFFF !important;
        border: 1px solid var(--border) !important;
        border-radius: 12px !important;
    }

    div[data-baseweb="select"] span,
    div[data-baseweb="select"] input,
    div[data-baseweb="select"] svg {
        color: var(--text) !important;
        fill: var(--text) !important;
    }

    /* BaseWeb may render the selected value as a nested div rather than a span. */
    div[data-testid="stSelectbox"] div[data-baseweb="select"] [role="combobox"],
    div[data-testid="stSelectbox"] div[data-baseweb="select"] [role="combobox"] *,
    div[data-testid="stSelectbox"] div[data-baseweb="select"] > div > div,
    div[data-testid="stSelectbox"] div[data-baseweb="select"] > div > div * {
        color: var(--text) !important;
        -webkit-text-fill-color: var(--text) !important;
        opacity: 1 !important;
    }

    div[data-testid="stSelectbox"] div[data-baseweb="select"] svg {
        color: var(--text) !important;
        fill: var(--text) !important;
        opacity: 1 !important;
    }

    div[data-baseweb="popover"] {
        z-index: 1000 !important;
    }

    div[data-baseweb="popover"] [role="listbox"],
    ul[role="listbox"] {
        background: #FFFFFF !important;
        border: 1px solid var(--border) !important;
        border-radius: 12px !important;
        box-shadow: 0 12px 30px rgba(15, 23, 42, 0.18) !important;
    }

    [role="option"],
    [role="option"] * {
        background: #FFFFFF !important;
        color: var(--text) !important;
    }

    [role="option"]:hover,
    [role="option"][aria-selected="true"] {
        background: #EAF3F7 !important;
    }

    div[data-testid="stMultiSelect"] span[data-baseweb="tag"] {
        background: #EAF3F7 !important;
        color: var(--text) !important;
        border-radius: 10px !important;
        font-weight: 600;
    }

    div[data-testid="stMultiSelect"] span[data-baseweb="tag"] button,
    div[data-testid="stMultiSelect"] span[data-baseweb="tag"] button * {
        color: var(--coral) !important;
        fill: var(--coral) !important;
        opacity: 1 !important;
    }

    div[data-testid="stExpander"] {
        background: #FFFFFF;
        border: 1px solid var(--border);
        border-radius: 14px;
    }

    div[data-testid="stExpander"] details > summary {
        min-height: 48px;
        background: #F7FBFD !important;
        border-radius: 13px;
    }

    div[data-testid="stExpander"] details[open] > summary {
        border-bottom: 1px solid var(--border);
        border-radius: 13px 13px 0 0;
    }

    div[data-testid="stExpander"] details > summary,
    div[data-testid="stExpander"] details > summary *,
    div[data-testid="stExpander"] details > summary svg {
        color: var(--navy) !important;
        fill: var(--navy) !important;
        opacity: 1 !important;
        font-weight: 700 !important;
    }

    div[data-testid="stExpander"] div[data-testid="stExpanderDetails"] {
        background: #FFFFFF !important;
    }

    div[data-testid="stExpander"] div[data-testid="stExpanderDetails"] *,
    [data-testid="stDataFrame"] *,
    [data-testid="stTable"] * {
        color: var(--text) !important;
    }

    @media (max-width: 768px) {
        div[data-testid="stAppViewBlockContainer"],
        .block-container {
            padding: 2.95rem 1rem 2rem;
        }

        .hero {
            padding: 1.25rem;
            border-radius: 14px;
        }

        .hero p,
        .plot-explainer-text,
        .indicator-card-meta {
            font-size: 1rem;
        }

        .indicator-card {
            padding: 1.1rem;
        }
    }

    @media (prefers-reduced-motion: reduce) {
        *, *::before, *::after {
            scroll-behavior: auto !important;
            transition-duration: 0.01ms !important;
            animation-duration: 0.01ms !important;
            animation-iteration-count: 1 !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)
# ---------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------

st.sidebar.title("Development Intelligence")
st.sidebar.caption("UNDP Timor-Leste NHDR 2018")

page = st.sidebar.radio(
    "Dashboard section",
    (
        "Overview",
        "Themes",
        "Indicators and trends",
        "Strengths and challenges",
        "Model evaluation",
    ),
)

st.sidebar.divider()

run_manifest = read_json(RESULTS / "run_manifest.json", {})

if run_manifest:
    st.sidebar.success("By : Owen Nda Diche")
    st.sidebar.caption(
        f"Extractor: {run_manifest.get('extractor_model')}  \n"
        f"Evaluator: {run_manifest.get('evaluator_model')}"
    )
else:
    st.sidebar.warning("Showing deterministic preparation outputs; local-LLM run pending.")


# ---------------------------------------------------------------------
# Shared header
# ---------------------------------------------------------------------

st.markdown(
    """
    <section class="hero">
      <h1>Timor-Leste Development Intelligence</h1>
      <p>Evidence-grounded analysis of the 2018 National Human Development Report,
      combining local language models with transparent PDF and table extraction.</p>
    </section>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------
# Load all data once
# ---------------------------------------------------------------------

theme_totals, theme_map = theme_frames()
indicators = indicators_frame()
time_series = time_series_frame()
evaluations = evaluation_frame()
comparison = pd.DataFrame(read_json(RESULTS / "model_comparison.json", {}).get("metrics", []))
strengths_payload = read_json(RESULTS / "strengths_challenges.json", {})


# ---------------------------------------------------------------------
# Page 1: Overview
# ---------------------------------------------------------------------

if page == "Overview":
    manifest = read_json(ROOT / "outputs" / "processed" / "manifest.json", {})

    columns = st.columns(4)
    columns[0].metric("PDF pages", manifest.get("page_count", "-"))
    columns[1].metric("Report sections", manifest.get("section_count", "-"))
    columns[2].metric("Evidence chunks", manifest.get("chunk_count", "-"))
    columns[3].metric("Extracted tables", manifest.get("table_count", "-"))

    st.markdown("## Executive summary")
    st.caption(
        "Key findings extracted from the report with source-page traceability. "
        "These findings summarise the main development signals identified by "
        "the local LLM pipeline."
    )

    findings = read_json(RESULTS / "report_findings.json", {}).get("key_results", [])

    if findings:
        for start in range(0, len(findings), 2):
            card_columns = st.columns(2)

            for offset, column in enumerate(card_columns):
                index = start + offset

                if index >= len(findings):
                    continue

                finding = findings[index]

                finding_text = clean_text(
                    finding.get("finding", ""),
                    "No finding text available."
                )

                source_pages = finding.get("source_pages", [])

                with column:
                    with st.container(border=True):
                        header_cols = st.columns([0.16, 0.84])

                        with header_cols[0]:
                            st.markdown(
                                f'<div class="finding-number">{index + 1}</div>',
                                unsafe_allow_html=True,
                            )

                        with header_cols[1]:
                            st.markdown(
                                '<div class="finding-label">Key finding</div>',
                                unsafe_allow_html=True,
                            )

                        st.markdown(
                            f'<div class="finding-text">{html.escape(finding_text)}</div>',
                            unsafe_allow_html=True,
                        )

                        if source_pages:
                            page_chips = "".join(
                                f'<span class="source-chip">p. {html.escape(str(page_number))}</span>'
                                for page_number in source_pages
                            )

                            st.markdown(
                                f'<div class="source-line">Source pages: {page_chips}</div>',
                                unsafe_allow_html=True,
                            )
                        else:
                            st.markdown(
                                '<div class="source-line">Source pages: not listed</div>',
                                unsafe_allow_html=True,
                            )

    else:
        st.markdown(
            """
            <div class="status-note">
                Key findings will appear after the Ollama analysis run.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.divider()

    st.markdown("## What to explore next")

    next_columns = st.columns(4)

    with next_columns[0]:
        with st.container(border=True):
            st.markdown(
                '<div class="next-card-title">Themes</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                """
                <div class="next-card-text">
                    View how education, employment, health, economy, gender,
                    climate, and inequality appear across the report.
                </div>
                """,
                unsafe_allow_html=True,
            )

    with next_columns[1]:
        with st.container(border=True):
            st.markdown(
                '<div class="next-card-title">Indicators</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                """
                <div class="next-card-text">
                    Inspect extracted numerical indicators, source evidence,
                    and reported time-based trends.
                </div>
                """,
                unsafe_allow_html=True,
            )

    with next_columns[2]:
        with st.container(border=True):
            st.markdown(
                '<div class="next-card-title">Strengths and challenges</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                """
                <div class="next-card-text">
                    Compare evidence-backed development progress, opportunities,
                    constraints, and source pages.
                </div>
                """,
                unsafe_allow_html=True,
            )

    with next_columns[3]:
        with st.container(border=True):
            st.markdown(
                '<div class="next-card-title">Model evaluation</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                """
                <div class="next-card-text">
                    Review evaluator scores for completeness, consistency,
                    factual alignment, and format compliance.
                </div>
                """,
                unsafe_allow_html=True,
            )

# ---------------------------------------------------------------------
# Page 2: Themes
# ---------------------------------------------------------------------

elif page == "Themes":
    st.markdown(
        """
        <div class="plot-explainer">
            <div class="plot-explainer-title">What this section shows</div>
            <div class="plot-explainer-text">
                This section helps you understand <b>which development themes appear most often</b>
                in the report and <b>where they are discussed</b> across the document.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.subheader("Thematic distribution")
    
    
  
   

    st.markdown(
        """
        <div class="plot-explainer">
            <div class="plot-explainer-title">Theme occurrence counts</div>
            <div class="plot-explainer-text">
                This chart compares how frequently each development theme appears across the
                extracted report chunks. Higher counts suggest that a theme receives more
                discussion or emphasis in the report.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if theme_totals.empty:
        st.info("Run the preparation stage to create thematic data.")
    else:
        figure = px.bar(
            theme_totals.sort_values("count"),
            x="count",
            y="theme",
            orientation="h",
            color="theme",
            color_discrete_map=THEME_COLORS,
            text="count",
        )

        figure.update_traces(
            textposition="outside",
            cliponaxis=False,
            textfont={"size": 15, "color": COLORS["text"]},
        )

        chart_layout(
            figure,
            "Theme occurrence counts",
            x_title="Number of report chunks",
            y_title="Development theme",
            height=540,
            show_legend=False,
        )

        show_plot(figure)

    st.subheader("Narrative map")

    st.markdown(
        """
        <div class="plot-explainer">
            <div class="plot-explainer-title">Where themes appear in the report</div>
            <div class="plot-explainer-text">
                This narrative map shows the <b>distribution of selected themes across PDF pages</b>.
                Each point marks a place where a theme was detected, helping you see how themes
                are spread through the report and which sections they appear in.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not theme_map.empty:
        selected_themes = st.multiselect(
            "Themes to display",
            sorted(theme_map["theme"].unique()),
            default=sorted(theme_map["theme"].unique()),
        )

        filtered = theme_map[theme_map["theme"].isin(selected_themes)].copy()

        if filtered.empty:
            st.info("Select at least one theme to display the narrative map.")
        else:
            figure = px.scatter(
                filtered,
                x="pdf_page",
                y="theme",
                color="theme",
                symbol="section",
                hover_data={
                    "evidence": True,
                    "section": True,
                    "pdf_page": True,
                    "theme": False,
                },
                color_discrete_map=THEME_COLORS,
            )

            figure.update_traces(
                marker={
                    "size": 13,
                    "line": {"width": 1, "color": "white"},
                }
            )

            chart_layout(
                figure,
                "Where themes appear in the report",
                x_title="PDF page number",
                y_title="Theme",
                height=580,
                show_legend=False,
            )

            figure.update_xaxes(dtick=20)
            show_plot(figure)

            st.caption(
                "Each point shows where a selected development theme was detected in the report. "
                "Hover over a point to view the section and supporting evidence."
            )

            with st.expander("Accessible narrative-map table"):
                st.dataframe(filtered, hide_index=True, width="stretch")
    else:
        st.info("Narrative map data is not available yet.")

# ---------------------------------------------------------------------
# Page 3: Indicators and trends
# ---------------------------------------------------------------------

elif page == "Indicators and trends":
    st.subheader("Core indicators")

    st.markdown(
        """
        <div class="plot-explainer">
            <div class="plot-explainer-title">Core indicators</div>
            <div class="plot-explainer-text">
                This section shows the <b>main numerical indicators extracted from the report</b>.
                Select an indicator to view its reported value, unit, year, population group,
                source page, and supporting evidence from the PDF.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    

    if indicators.empty:
        st.info("Core indicators will appear after the local-LLM extraction run.")
    else:
        indicator_view = indicators.copy()
        indicator_evaluation = evaluations[evaluations["output"] == "Indicators"]
        indicator_score = (
            indicator_evaluation["score"].mean() if not indicator_evaluation.empty else None
        )
        indicator_verdict = (
            clean_text(indicator_evaluation["verdict"].iloc[0], "not evaluated")
            if not indicator_evaluation.empty
            else "not evaluated"
        )
        indicator_view["extractor_model"] = run_manifest.get("extractor_model", "not recorded")
        indicator_view["evaluator_verdict"] = indicator_verdict
        indicator_view["evaluator_mean_score"] = (
            f"{indicator_score:.2f} / 5" if indicator_score is not None else "not evaluated"
        )

        for column in ["name", "unit", "year", "population_group", "pdf_page", "evidence"]:
            if column not in indicator_view.columns:
                indicator_view[column] = ""

        indicator_view["numeric_value"] = pd.to_numeric(
            indicator_view["numeric_value"],
            errors="coerce",
        )
        indicator_view = indicator_view[indicator_view["numeric_value"].notna()].copy()

        if indicator_view.empty:
            st.info("No numeric indicator values are currently available.")
        else:
            indicator_view["display_value"] = indicator_view.apply(
                lambda row: (
                    f"{row['numeric_value']:.3f}"
                    if clean_text(row.get("name")).lower() == "hdi value"
                    else format_number(row["numeric_value"])
                ),
                axis=1,
            )
            indicator_view["display_unit"] = indicator_view["unit"].apply(
                lambda value: clean_text(value, "reported units")
            )
            indicator_view["display_year"] = indicator_view["year"].apply(
                lambda value: clean_text(value, "No year reported")
            )
            indicator_view["display_group"] = indicator_view["population_group"].apply(
                lambda value: clean_text(value, "All population")
            )
            indicator_view["display_page"] = indicator_view["pdf_page"].apply(
                lambda value: clean_text(value, "No page reported")
            )

            names = sorted(indicator_view["name"].dropna().unique())
            selected = st.selectbox("Indicator", names)
            selected_frame = indicator_view[indicator_view["name"] == selected].copy()

            st.markdown(f"#### {selected}")

            if selected_frame.empty:
                st.info("No values available for the selected indicator.")
            else:
                value_count = len(selected_frame)

                if value_count == 1:
                    row = selected_frame.iloc[0]

                    value_text = html.escape(row["display_value"])
                    unit_text = html.escape(row["display_unit"])
                    year_text = html.escape(row["display_year"])
                    group_text = html.escape(row["display_group"])
                    page_text = html.escape(row["display_page"])
                    section_text = html.escape(clean_text(row.get("source_section"), "Not identified"))
                    extractor_text = html.escape(row["extractor_model"])
                    validation_text = html.escape(
                        clean_text(row.get("validation_method"), "not recorded").replace("_", "-")
                    )
                    evaluation_text = html.escape(
                        f"{row['evaluator_mean_score']} ({row['evaluator_verdict']})"
                    )

                    st.markdown(
                        f"""
                        <div class="indicator-card">
                            <div class="indicator-card-label">Extracted indicator value</div>
                            <div class="indicator-card-value">{value_text} {unit_text}</div>
                            <div class="indicator-card-meta">
                                <strong>Year:</strong> {year_text}<br>
                                <strong>Population group:</strong> {group_text}<br>
                                <strong>Source:</strong> PDF page {page_text}<br>
                                <strong>Report section:</strong> {section_text}<br>
                                <strong>Raw extraction model:</strong> {extractor_text}<br>
                                <strong>Displayed-value validation:</strong> {validation_text}<br>
                                <strong>Overall indicator-set evaluation:</strong> {evaluation_text}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    st.caption(
                        "Only one value was extracted for this indicator, so it is shown as a metric card "
                        "rather than a bar chart. A bar chart is more useful when comparing multiple years, "
                        "groups, countries, or models."
                    )

                    with st.expander("Source evidence"):
                        evidence = clean_text(
                            row.get("evidence", ""),
                            "No evidence snippet available for this value.",
                        )
                        st.write(evidence)

                else:
                    selected_frame["chart_label"] = (
                        selected_frame["display_year"]
                        + " | "
                        + selected_frame["display_group"]
                    )
                    selected_frame["chart_label"] = selected_frame["chart_label"].apply(
                        lambda value: shorten_label(value, 55)
                    )

                    show_indicator_legend = selected_frame["display_group"].nunique() > 1

                    figure = px.bar(
                        selected_frame,
                        x="chart_label",
                        y="numeric_value",
                        color="display_group",
                        text="numeric_value",
                        color_discrete_sequence=[
                            COLORS["blue"],
                            COLORS["gold"],
                            COLORS["coral"],
                            COLORS["teal"],
                        ],
                        hover_data={
                            "display_unit": True,
                            "display_page": True,
                            "evidence": True,
                            "chart_label": False,
                            "numeric_value": ":.2f",
                        },
                    )

                    figure.update_traces(
                        texttemplate="%{text:.2f}",
                        textposition="outside",
                        cliponaxis=False,
                        textfont={"size": 14, "color": COLORS["text"]},
                    )

                    unit_label = selected_frame["display_unit"].iloc[0]

                    chart_layout(
                        figure,
                        f"Comparison of reported values: {selected}",
                        x_title="Year / population group",
                        y_title=unit_label,
                        height=540,
                        bottom_margin=130,
                        show_legend=show_indicator_legend,
                    )

                    figure.update_xaxes(tickangle=-25)
                    show_plot(figure)

                    with st.expander("Source evidence table"):
                        st.dataframe(
                            selected_frame[
                                [
                                    "name",
                                    "display_value",
                                    "display_unit",
                                    "display_year",
                                    "display_group",
                                    "display_page",
                                    "source_section",
                                    "extractor_model",
                                    "validation_method",
                                    "evaluator_mean_score",
                                    "evaluator_verdict",
                                    "evidence",
                                ]
                            ].rename(
                                columns={
                                    "name": "Indicator",
                                    "display_value": "Value",
                                    "display_unit": "Unit",
                                    "display_year": "Year",
                                    "display_group": "Population group",
                                    "display_page": "PDF page",
                                    "source_section": "Report section",
                                    "extractor_model": "Raw extraction model",
                                    "validation_method": "Displayed-value validation",
                                    "evaluator_mean_score": "Overall indicator-set mean score",
                                    "evaluator_verdict": "Overall indicator-set verdict",
                                    "evidence": "Evidence",
                                }
                            ),
                            hide_index=True,
                            width="stretch",
                        )

    st.subheader("Development trends")
    st.markdown(
        """
        <div class="plot-explainer">
            <div class="plot-explainer-title">Development trends</div>
            <div class="plot-explainer-text">
                This chart compares extracted time-series indicators across reported years.
                Use <b>indexed mode</b> when comparing indicators with different units, or
                switch to <b>raw values</b> to see the original reported figures.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(
        "Trend labels and values come from extracted report tables. Indexed mode supports "
        "comparison, but it does not validate the underlying extraction."
    )

    if time_series.empty:
        st.info("No time-series candidates are available yet.")
    else:
        time_series_clean = time_series.copy()

        required_trend_columns = ["series", "year_label", "numeric_value"]
        missing_trend_columns = [
            column for column in required_trend_columns if column not in time_series_clean.columns
        ]

        if missing_trend_columns:
            st.warning(
                "The time-series file is missing these required columns: "
                + ", ".join(missing_trend_columns)
            )
        else:
            time_series_clean["numeric_value"] = pd.to_numeric(
                time_series_clean["numeric_value"],
                errors="coerce",
            )
            time_series_clean = time_series_clean[time_series_clean["numeric_value"].notna()].copy()

            time_series_clean["series"] = (
                time_series_clean["series"].fillna("").astype(str).str.strip()
            )
            time_series_clean["year_label"] = (
                time_series_clean["year_label"].fillna("").astype(str).str.strip()
            )
            time_series_clean = time_series_clean[
                (time_series_clean["series"] != "")
                & (time_series_clean["year_label"] != "")
            ].copy()

            if time_series_clean.empty:
                st.info("No valid numeric time-series values are available yet.")
            else:
                if "indexed_value" not in time_series_clean.columns:
                    time_series_clean["indexed_value"] = time_series_clean.groupby("series")[
                        "numeric_value"
                    ].transform(
                        lambda values: (values / values.iloc[0] * 100)
                        if values.iloc[0] != 0
                        else values
                    )

                time_series_clean["indexed_value"] = pd.to_numeric(
                    time_series_clean["indexed_value"],
                    errors="coerce",
                )

                family_order = [
                    family
                    for family in [
                        "Education outcomes",
                        "Budget shares",
                        "Budget totals",
                        "Population projections",
                    ]
                    if family in set(time_series_clean["trend_family"])
                ]
                selected_family = st.selectbox(
                    "Trend family",
                    family_order,
                    help="Related indicators are grouped so incompatible units and projections are not mixed.",
                )
                family_trends = time_series_clean[
                    time_series_clean["trend_family"] == selected_family
                ].copy()
                series_options = sorted(family_trends["series"].dropna().unique())
                preferred_series = "Adult literacy, ages 15+, %"
                default_series = [
                    preferred_series if preferred_series in series_options else series_options[0]
                ]

                selected_series = st.multiselect(
                    "Choose series to compare",
                    series_options,
                    default=default_series,
                    key=f"trend_series_{selected_family}",
                    help=(
                        "Select one or more extracted time-series indicators. "
                        "Indexed mode is best when the series have different units."
                    ),
                )

                filtered_trends = family_trends[
                    family_trends["series"].isin(selected_series)
                ].copy()

                if filtered_trends.empty:
                    st.info("Select at least one time-series value to display the trend chart.")
                else:
                    chart_mode = st.radio(
                        "Trend chart scale",
                        (
                            "Indexed values, first available year = 100",
                            "Raw reported values",
                        ),
                        horizontal=True,
                        index=1 if len(selected_series) == 1 else 0,
                        key=f"trend_scale_{selected_family}",
                    )

                    if chart_mode.startswith("Indexed"):
                        y_column = "indexed_value"
                        y_title = "Index value, first available year = 100"
                        chart_note = (
                            "Indexed mode is useful because the selected series may use different units. "
                            "Each line starts from 100, so the chart focuses on the direction of change "
                            "rather than the original scale."
                        )
                        filtered_trends = filtered_trends[
                            filtered_trends["indexed_value"].notna()
                        ].copy()
                    else:
                        y_column = "numeric_value"
                        y_title = "Reported indicator value"
                        chart_note = (
                            "Raw mode shows the extracted values as reported. Be careful when comparing "
                            "series with different units."
                        )

                    if filtered_trends.empty:
                        st.info("The selected series cannot be shown in this chart mode.")
                    else:
                        if "unit" not in filtered_trends.columns:
                            filtered_trends["unit"] = ""
                        if "pdf_page" not in filtered_trends.columns:
                            filtered_trends["pdf_page"] = ""

                        filtered_trends["series_short"] = filtered_trends["series"].apply(
                            lambda value: shorten_label(value, 65)
                        )

                        filtered_trends = filtered_trends.dropna(subset=["year_position"])
                        filtered_trends = filtered_trends.sort_values(
                            by=["series", "year_position", "year_label"]
                        )
                        period_labels = (
                            filtered_trends[["year_position", "year_label"]]
                            .drop_duplicates(subset=["year_position"])
                            .sort_values("year_position")
                        )
                        use_small_multiples = len(selected_series) > 3
                        if use_small_multiples:
                            st.info(
                                "More than three series are selected, so the dashboard has switched "
                                "to small multiples to keep every trend readable."
                            )

                        figure = px.line(
                            filtered_trends,
                            x="year_position",
                            y=y_column,
                            color="series_short",
                            facet_row="series_short" if use_small_multiples else None,
                            facet_row_spacing=0.045 if use_small_multiples else 0.0,
                            markers=True,
                            hover_data={
                                "series_short": False,
                                "series": True,
                                "unit": True,
                                "pdf_page": True,
                                "numeric_value": ":.2f",
                                "indexed_value": ":.2f",
                                "year_label": True,
                                "year_position": False,
                            },
                            color_discrete_sequence=TREND_COLORS,
                            labels={
                                "year_position": "Reported year or period",
                                "year_label": "Reported period",
                                y_column: y_title,
                                "series_short": "Series",
                                "series": "Full series name",
                                "unit": "Unit",
                                "pdf_page": "PDF page",
                            },
                        )

                        figure.update_traces(line={"width": 3}, marker={"size": 9})
                        if selected_family == "Population projections":
                            figure.update_traces(line={"width": 3, "dash": "dash"})
                        if use_small_multiples:
                            figure.update_yaxes(matches=None)
                            figure.for_each_annotation(
                                lambda annotation: annotation.update(
                                    text=annotation.text.split("=")[-1],
                                    font={"size": 14, "color": COLORS["navy"]},
                                )
                            )

                        chart_layout(
                            figure,
                            f"Reported change over time — {selected_family}",
                            x_title="Reported year or period",
                            y_title=y_title,
                            height=(max(620, 225 * len(selected_series)) if use_small_multiples else 580),
                            left_margin=95,
                            right_margin=80,
                            bottom_margin=150,
                            show_legend=not use_small_multiples,
                        )

                        figure.update_xaxes(
                            type="linear",
                            tickvals=period_labels["year_position"].tolist(),
                            ticktext=period_labels["year_label"].tolist(),
                            tickangle=-35,
                        )
                        if not use_small_multiples:
                            figure.update_layout(
                                legend={
                                    "orientation": "h",
                                    "yanchor": "top",
                                    "y": -0.28,
                                    "xanchor": "left",
                                    "x": 0,
                                }
                            )
                        show_plot(figure)

                        st.markdown(
                            f"""
                            <div style="
                                background:#ffffff;
                                border:1px solid #d8e1e8;
                                border-radius:14px;
                                padding:1rem 1.2rem;
                                margin-top:1rem;
                                margin-bottom:1.2rem;
                                color:#102033;
                                line-height:1.7;
                            ">
                                <b>How to read this chart:</b> {html.escape(chart_note)}
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                        with st.expander("Accessible time-series table"):
                            st.dataframe(
                                filtered_trends,
                                hide_index=True,
                                width="stretch",
                            )


# ---------------------------------------------------------------------
# Page 4: Strengths and challenges
# ---------------------------------------------------------------------

elif page == "Strengths and challenges":
    st.subheader("Development strengths and challenges")
    st.markdown(
        """
        <div class="plot-explainer">
            <div class="plot-explainer-title">Evidence-backed qualitative findings</div>
            <div class="plot-explainer-text">
                These cards distinguish documented development strengths from continuing
                constraints. Every item retains exact PDF-page provenance,
                a short evidence summary, and model lineage.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    strengths = strengths_payload.get("strengths", [])
    challenges = strengths_payload.get("challenges", [])
    qualitative_evaluation = evaluations[evaluations["output"] == "Strengths Challenges"]
    qualitative_score = (
        qualitative_evaluation["score"].mean() if not qualitative_evaluation.empty else None
    )
    qualitative_verdict = (
        qualitative_evaluation["verdict"].iloc[0]
        if not qualitative_evaluation.empty
        else "not evaluated"
    )

    summary_columns = st.columns(4)
    summary_columns[0].metric("Strengths", len(strengths))
    summary_columns[1].metric("Challenges", len(challenges))
    summary_columns[2].metric(
        "Overall set score",
        f"{qualitative_score:.2f} / 5" if qualitative_score is not None else "—",
    )
    summary_columns[3].metric("Overall set verdict", readable_name(qualitative_verdict))

    card_columns = st.columns(2)
    groups = (
        ("Key strengths", strengths, COLORS["teal"]),
        ("Key challenges", challenges, COLORS["coral"]),
    )
    accessible_rows = []

    for column, (heading, items, accent) in zip(card_columns, groups):
        with column:
            st.markdown(f"### {heading}")
            for item in items:
                title = html.escape(clean_text(item.get("item"), "Untitled finding"))
                explanation = html.escape(
                    clean_text(item.get("explanation"), "No explanation supplied.")
                )
                classification = html.escape(
                    clean_text(item.get("classification"), "Finding")
                )
                evidence = html.escape(
                    clean_text(item.get("evidence"), "No evidence excerpt supplied.")
                )
                pages = item.get("source_pages", [])
                page_chips = "".join(
                    f'<span class="source-chip">p. {html.escape(str(page_number))}</span>'
                    for page_number in pages
                )
                with st.container(border=True):
                    st.markdown(
                        f"""
                        <div style="border-left:5px solid {accent}; padding-left:0.9rem;">
                            <div class="next-card-title">{title}</div>
                            <div class="source-line">Classification: {classification}</div>
                            <div class="next-card-text">{explanation}</div>
                            <div class="source-line"><strong>Evidence summary:</strong> {evidence}</div>
                            <div class="source-line">Source pages: {page_chips}</div>
                            <div class="source-line">Generated by: {html.escape(run_manifest.get('extractor_model', 'not recorded'))}</div>
                            <div class="source-line">Validation: source-verified</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                accessible_rows.append(
                    {
                        "Type": heading.removeprefix("Key ").title(),
                        "Finding": clean_text(item.get("item")),
                        "Classification": clean_text(item.get("classification")),
                        "Explanation": clean_text(item.get("explanation")),
                        "Evidence summary": clean_text(item.get("evidence")),
                        "PDF pages": ", ".join(str(page) for page in pages),
                        "Generated by": run_manifest.get("extractor_model", "not recorded"),
                        "Validation": "source-verified",
                        "Overall set evaluator score": qualitative_score,
                        "Overall set evaluator verdict": qualitative_verdict,
                    }
                )

    with st.expander("Accessible strengths and challenges table"):
        st.dataframe(pd.DataFrame(accessible_rows), hide_index=True, width="stretch")


# ---------------------------------------------------------------------
# Page 5: Model evaluation
# ---------------------------------------------------------------------

else:
    st.subheader("Independent evaluator scores")
    st.markdown(
        """
        <div class="plot-explainer">
            <div class="plot-explainer-title">Independent evaluator scores</div>
            <div class="plot-explainer-text">
                This section shows how the extracted outputs were judged by a separate local model.
                The scores help assess <b>completeness, consistency, factual alignment, and format quality</b>
                across the main outputs generated by the pipeline.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if evaluations.empty:
        st.info("Evaluator scores will appear after Qwen outputs are assessed by Llama 3.2.")
    else:
        st.warning(
            "Evaluator scores are model-generated judgments, not ground truth. Read the "
            "justification and source evidence before accepting a low or high score."
        )
        pivot = evaluations.pivot_table(
            index="output",
            columns="criterion",
            values="score",
            aggfunc="mean",
        )

        heatmap = px.imshow(
            pivot,
            text_auto=".1f",
            zmin=1,
            zmax=5,
            color_continuous_scale=[
                [0, COLORS["coral"]],
                [0.5, "#F2D06B"],
                [1, COLORS["teal"]],
            ],
            aspect="auto",
        )

        chart_layout(
            heatmap,
            "Evaluation matrix: 1 poor to 5 excellent",
            x_title="Evaluation criterion",
            y_title="Extracted output",
            height=540,
            left_margin=150,
            bottom_margin=130,
        )

        heatmap.update_xaxes(tickangle=-20)
        heatmap.update_traces(
            textfont={"size": 14},
            hovertemplate=(
                "Output: %{y}<br>"
                "Criterion: %{x}<br>"
                "Score: %{z:.1f}<extra></extra>"
            ),
        )

        show_plot(heatmap)

        means = evaluations.groupby("criterion", as_index=False)["score"].mean()
        means = means.dropna(subset=["score"])

        if not means.empty:
            radar = go.Figure(
                go.Scatterpolar(
                    r=means["score"].tolist() + [means["score"].iloc[0]],
                    theta=means["criterion"].tolist() + [means["criterion"].iloc[0]],
                    fill="toself",
                    name="Mean evaluator score",
                    line={"color": COLORS["blue"], "width": 3},
                    fillcolor="rgba(23,107,135,0.22)",
                )
            )

            radar.update_polars(
                radialaxis={
                    "range": [0, 5],
                    "tickvals": [1, 2, 3, 4, 5],
                    "tickfont": {"size": 13, "color": COLORS["text"]},
                    "gridcolor": COLORS["grid"],
                },
                angularaxis={
                    "tickfont": {"size": 15, "color": COLORS["text"]},
                },
            )

            chart_layout(
                radar,
                "Average model-generated quality profile across evaluation criteria",
                height=560,
                show_legend=False,
                left_margin=80,
                right_margin=80,
                bottom_margin=70,
            )

            show_plot(radar)

        evaluation_notes = evaluations[
            [
                "output",
                "verdict",
                "model_verdict",
                "verdict_policy_applied",
                "policy_reason",
                "justification",
                "unsupported_claims",
                "missing_items",
            ]
        ].drop_duplicates()

        with st.expander("Evaluator explanations and limitations"):
            st.dataframe(evaluation_notes, hide_index=True, width="stretch")

        with st.expander("Accessible evaluation score table"):
            st.dataframe(
                evaluations[["output", "criterion", "score", "verdict"]],
                hide_index=True,
                width="stretch",
            )

    st.subheader("Cross-model behaviour")
    st.markdown(
        """
        <div class="plot-explainer">
            <div class="plot-explainer-title">Cross-model behaviour</div>
            <div class="plot-explainer-text">
                This section compares how different local language models behave on the same task.
                It highlights the trade-off between <b>speed, output quality, JSON reliability,
                and run-to-run stability</b>.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if comparison.empty:
        st.info("Run the optional compare command to benchmark Qwen, Llama, and Phi-3.")
    else:
        plot_frame = comparison.dropna(subset=["mean_judge_score"]).copy()

        if {
            "valid_response_count",
            "total_response_count",
            "evaluated_response_count",
            "judge_models",
        }.issubset(comparison.columns):
            coverage = "; ".join(
                f"{row.model}: {int(row.valid_response_count)}/{int(row.total_response_count)} "
                f"valid responses, {int(row.evaluated_response_count)} judged by "
                f"{row.judge_models or 'not available'}"
                for row in comparison.itertuples()
            )
            st.caption(f"Benchmark coverage — {coverage}.")
            st.warning(
                "Judge scores are not perfectly judge-identical: Qwen and Phi-3 are judged "
                "by Llama 3.2, while Llama 3.2 is judged by Qwen to avoid self-evaluation."
            )

        if not plot_frame.empty:
            figure = px.scatter(
                plot_frame,
                x="average_seconds",
                y="mean_judge_score",
                size="json_valid_rate",
                color="model",
                text="model",
                hover_data=[
                    column
                    for column in [
                        "json_valid_rate",
                        "stability_jaccard",
                        "average_indicator_count",
                        "average_word_count",
                        "missing_field_rate",
                        "factual_alignment_score",
                        "valid_response_count",
                        "total_response_count",
                        "evaluated_response_count",
                        "judge_models",
                    ]
                    if column in plot_frame.columns
                ],
                color_discrete_sequence=[
                    COLORS["blue"],
                    COLORS["teal"],
                    COLORS["gold"],
                ],
            )

            figure.update_traces(
                textposition="top center",
                marker={"line": {"width": 1, "color": "white"}},
            )

            chart_layout(
                figure,
                "Quality-speed trade-off across local models",
                x_title="Mean generation time in seconds, lower is better",
                y_title="Mean judge quality score, 1 to 5",
                height=540,
            )

            figure.update_yaxes(range=[0, 5.2])
            show_plot(figure)
            st.caption(
                "Bubble size represents the valid-JSON rate. Quality scores include only "
                "responses that produced valid JSON and could be judged."
            )

        richness_columns = [
            column
            for column in ["average_theme_count", "average_indicator_count"]
            if column in comparison.columns
        ]
        if richness_columns:
            richness = comparison.melt(
                id_vars="model",
                value_vars=richness_columns,
                var_name="metric",
                value_name="average_count",
            )
            richness["metric"] = richness["metric"].map(
                {
                    "average_theme_count": "Themes extracted",
                    "average_indicator_count": "Numerical facts extracted",
                }
            ).fillna(richness["metric"])
            figure = px.bar(
                richness,
                x="model",
                y="average_count",
                color="metric",
                barmode="group",
                text_auto=".2f",
                color_discrete_sequence=[COLORS["blue"], COLORS["gold"]],
            )
            chart_layout(
                figure,
                "Output richness among valid JSON responses",
                x_title="Local LLM",
                y_title="Mean items extracted per valid response",
                height=520,
            )
            show_plot(figure)
            st.caption(
                "Richness measures output quantity, not factual quality, and excludes invalid "
                "JSON responses."
            )

        metric_columns = [
            column
            for column in ["json_valid_rate", "stability_jaccard"]
            if column in comparison.columns
        ]

        if metric_columns:
            stability = comparison.melt(
                id_vars="model",
                value_vars=metric_columns,
                var_name="metric",
                value_name="score",
            )

            stability["metric"] = stability["metric"].map(
                {
                    "json_valid_rate": "Valid JSON rate",
                    "stability_jaccard": "Run-to-run stability",
                }
            ).fillna(stability["metric"])

            figure = px.bar(
                stability,
                x="model",
                y="score",
                color="metric",
                barmode="group",
                text_auto=".2f",
                color_discrete_sequence=[COLORS["blue"], COLORS["gold"]],
            )

            figure.update_traces(
                textposition="outside",
                cliponaxis=False,
                textfont={"size": 14, "color": COLORS["text"]},
            )

            chart_layout(
                figure,
                "Structured-output reliability and run-to-run stability",
                x_title="Local LLM",
                y_title="Proportion, 0 to 1",
                height=540,
            )

            figure.update_yaxes(range=[0, 1.08])
            show_plot(figure)
            unavailable = comparison.loc[
                comparison["stability_jaccard"].isna(), "model"
            ].tolist()
            if unavailable:
                st.caption(
                    "Run-to-run stability is unavailable for "
                    f"{', '.join(unavailable)} because no evidence chunk had enough repeated "
                    "valid responses for a pairwise comparison."
                )

        with st.expander("Accessible comparison table"):
            st.dataframe(comparison, hide_index=True, width="stretch")
