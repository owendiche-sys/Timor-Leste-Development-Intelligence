"""JSON Schemas passed directly to Ollama for constrained structured output."""

from __future__ import annotations


def object_schema(properties: dict, required: list[str]) -> dict:
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


PAGE = {"type": "integer", "minimum": 1, "maximum": 160}
SHORT_TEXT = {"type": "string", "maxLength": 220}

FACT = object_schema(
    {"claim": SHORT_TEXT, "pdf_page": PAGE, "evidence": {"type": "string", "maxLength": 180}},
    ["claim", "pdf_page", "evidence"],
)
NUMBER = object_schema(
    {
        "label": {"type": "string", "maxLength": 80},
        "value": {"type": ["string", "number"]},
        "year": {"type": ["string", "integer", "null"]},
        "pdf_page": PAGE,
    },
    ["label", "value", "year", "pdf_page"],
)
THEME = object_schema(
    {
        "theme": {
            "type": "string",
            "enum": ["education", "health", "inequality", "economy", "gender", "climate", "employment"],
        },
        "evidence": {"type": "string", "maxLength": 140},
        "pdf_page": PAGE,
    },
    ["theme", "evidence", "pdf_page"],
)

CHUNK_ANALYSIS_SCHEMA = object_schema(
    {
        "key_facts": {"type": "array", "items": FACT, "minItems": 3, "maxItems": 3},
        "main_argument": SHORT_TEXT,
        "important_numbers": {"type": "array", "items": NUMBER, "maxItems": 3},
        "themes": {"type": "array", "items": THEME, "maxItems": 7},
    },
    ["key_facts", "main_argument", "important_numbers", "themes"],
)

CHAPTER_SUMMARY_SCHEMA = object_schema(
    {
        "chapter": {"type": "string", "maxLength": 140},
        "summary": {"type": "string", "maxLength": 850},
        "key_points": {"type": "array", "items": SHORT_TEXT, "minItems": 3, "maxItems": 5},
        "source_pages": {"type": "array", "items": PAGE, "minItems": 1, "maxItems": 12},
    },
    ["chapter", "summary", "key_points", "source_pages"],
)

REPORT_FINDING = object_schema(
    {
        "finding": SHORT_TEXT,
        "source_pages": {"type": "array", "items": PAGE, "minItems": 1, "maxItems": 6},
        "significance": SHORT_TEXT,
    },
    ["finding", "source_pages", "significance"],
)
REPORT_FINDINGS_SCHEMA = object_schema(
    {
        "report_title": {"type": "string", "maxLength": 180},
        "key_results": {"type": "array", "items": REPORT_FINDING, "minItems": 5, "maxItems": 6},
    },
    ["report_title", "key_results"],
)

INDICATOR = object_schema(
    {
        "name": {
            "type": "string",
            "enum": [
                "HDI value",
                "HDI rank",
                "Life expectancy (years)",
                "Expected years of schooling",
                "Mean years of schooling",
                "GNI per capita",
                "Population",
            ],
        },
        "value": {"type": ["number", "string"]},
        "unit": {"type": "string", "maxLength": 60},
        "year": {"type": ["integer", "string", "null"]},
        "population_group": {"type": ["string", "null"], "maxLength": 100},
        "pdf_page": PAGE,
        "evidence": {"type": "string", "maxLength": 220},
    },
    ["name", "value", "unit", "year", "population_group", "pdf_page", "evidence"],
)
INDICATORS_SCHEMA = object_schema(
    {
        "indicators": {"type": "array", "items": INDICATOR, "maxItems": 7},
        "not_found": {"type": "array", "items": {"type": "string", "maxLength": 100}, "maxItems": 7},
    },
    ["indicators", "not_found"],
)

STRENGTH_ITEM = object_schema(
    {
        "item": {"type": "string", "maxLength": 120},
        "explanation": SHORT_TEXT,
        "source_pages": {"type": "array", "items": PAGE, "minItems": 1, "maxItems": 5},
    },
    ["item", "explanation", "source_pages"],
)
STRENGTHS_CHALLENGES_SCHEMA = object_schema(
    {
        "strengths": {"type": "array", "items": STRENGTH_ITEM, "minItems": 5, "maxItems": 6},
        "challenges": {"type": "array", "items": STRENGTH_ITEM, "minItems": 5, "maxItems": 6},
    },
    ["strengths", "challenges"],
)

TREND_POINT = object_schema(
    {"year": {"type": ["string", "integer"]}, "value": {"type": "number"}},
    ["year", "value"],
)
TREND_SERIES = object_schema(
    {
        "series_name": {"type": "string", "maxLength": 140},
        "unit": {"type": "string", "maxLength": 60},
        "population_group": {"type": ["string", "null"], "maxLength": 100},
        "pdf_page": PAGE,
        "points": {"type": "array", "items": TREND_POINT, "minItems": 3, "maxItems": 12},
    },
    ["series_name", "unit", "population_group", "pdf_page", "points"],
)
TRENDS_SCHEMA = object_schema(
    {"series": {"type": "array", "items": TREND_SERIES, "maxItems": 12}},
    ["series"],
)

EVALUATION_SCHEMA = object_schema(
    {
        "task": {"type": "string", "maxLength": 140},
        "scores": object_schema(
            {
                "factual_alignment": {"type": "integer", "minimum": 1, "maximum": 5},
                "completeness": {"type": "integer", "minimum": 1, "maximum": 5},
                "consistency": {"type": "integer", "minimum": 1, "maximum": 5},
                "format_compliance": {"type": "integer", "minimum": 1, "maximum": 5},
            },
            ["factual_alignment", "completeness", "consistency", "format_compliance"],
        ),
        "unsupported_claims": {"type": "array", "items": SHORT_TEXT, "maxItems": 5},
        "missing_items": {"type": "array", "items": SHORT_TEXT, "maxItems": 5},
        "verdict": {"type": "string", "enum": ["pass", "revise"]},
        "justification": {"type": "string", "maxLength": 500},
    },
    ["task", "scores", "unsupported_claims", "missing_items", "verdict", "justification"],
)

COMPARISON_NUMBER = object_schema(
    {
        "label": {"type": "string", "maxLength": 80},
        "value": {"type": ["string", "number"]},
        "unit": {"type": "string", "maxLength": 60},
        "year": {"type": ["string", "integer", "null"]},
        "pdf_page": PAGE,
        "evidence": {"type": "string", "maxLength": 180},
    },
    ["label", "value", "unit", "year", "pdf_page", "evidence"],
)

COMPARISON_SCHEMA = object_schema(
    {
        "summary": {"type": "string", "maxLength": 650},
        "themes": {
            "type": "array",
            "items": {
                "type": "string",
                "enum": ["education", "health", "inequality", "economy", "gender", "climate", "employment"],
            },
            "maxItems": 7,
        },
        "numerical_facts": {"type": "array", "items": COMPARISON_NUMBER, "maxItems": 5},
    },
    ["summary", "themes", "numerical_facts"],
)


def theme_batch_schema(chunk_ids: list[str]) -> dict:
    item = object_schema(
        {
            "chunk_id": {"type": "string", "enum": chunk_ids},
            "themes": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": ["education", "health", "inequality", "economy", "gender", "climate", "employment"],
                },
                "uniqueItems": True,
                "maxItems": 7,
            },
        },
        ["chunk_id", "themes"],
    )
    return object_schema(
        {
            "classifications": {
                "type": "array",
                "items": item,
                "minItems": len(chunk_ids),
                "maxItems": len(chunk_ids),
            }
        },
        ["classifications"],
    )
