"""Versioned prompts used in the assignment pipeline.

Keeping prompts in code makes the experiment auditable and lets the final
report reproduce the exact instructions rather than paraphrasing them.
"""

from __future__ import annotations

import json
from typing import Iterable

from .config import THEMES


PROMPT_VERSION = "1.1.0"


def _source_block(text: str, start_page: int, end_page: int) -> str:
    return (
        f"<SOURCE pdf_pages=\"{start_page}-{end_page}\">\n"
        f"{text}\n"
        "</SOURCE>"
    )


def chunk_summary_prompt(chunk: dict) -> str:
    schema = {
        "key_facts": [
            {"claim": "fact stated in the source", "pdf_page": 1, "evidence": "short source phrase"}
        ],
        "main_argument": "one concise sentence",
        "important_numbers": [
            {"label": "indicator", "value": "source value", "year": "year or null", "pdf_page": 1}
        ],
        "themes": [
            {"theme": "education", "evidence": "maximum 12 source words", "pdf_page": 1}
        ],
    }
    return f"""You are extracting evidence from a UN development report.
Use only the source block. Do not use prior knowledge or infer missing values.
Capture exactly 3-5 decision-relevant facts and at most 3 important quantities.
For themes, include only directly present themes selected from {json.dumps(list(THEMES))};
omit absent themes. Keep every evidence phrase below 20 words.
Every factual item must cite a PDF page within {chunk['start_pdf_page']}-{chunk['end_pdf_page']}.
Return only valid JSON matching this example structure:
{json.dumps(schema, ensure_ascii=False)}

{_source_block(chunk['text'], chunk['start_pdf_page'], chunk['end_pdf_page'])}"""


def chapter_synthesis_prompt(section_title: str, notes: Iterable[dict]) -> str:
    schema = {
        "chapter": section_title,
        "summary": "fewer than 100 words",
        "key_points": ["three to five concise points"],
        "source_pages": [1, 2],
    }
    return f"""You are writing a faithful chapter summary of a UN development report.
Synthesize only the supplied evidence notes. Preserve qualifications and important numbers.
The summary must contain fewer than 100 words. Do not add external information.
Return only valid JSON matching this structure:
{json.dumps(schema, ensure_ascii=False)}

CHAPTER: {section_title}
EVIDENCE NOTES:
{json.dumps(list(notes), ensure_ascii=False)}"""


def chapter_source_summary_prompt(section_title: str, source_digest: str) -> str:
    schema = {
        "chapter": section_title,
        "summary": "fewer than 100 words",
        "key_points": ["three to five concise points"],
        "source_pages": [1, 2],
    }
    return f"""Summarize this chapter of a UN development report using only the source digest.
The digest is extractive and samples every chapter segment. Preserve important qualifications,
population groups, and statistical definitions exactly. Never relabel "without jobs", NEET,
dependency ratio, labour-force participation, or similar measures as "unemployment". Do not
attach a nearby number to a different indicator. The summary must contain fewer than 100 words. Page citations must come from
the bracketed PDF-page labels. Return only the required JSON object:
{json.dumps(schema, ensure_ascii=False)}

CHAPTER: {section_title}
SOURCE DIGEST:
{source_digest}"""


def batch_themes_prompt(items: list[dict]) -> str:
    compact = [
        {
            "chunk_id": item["chunk_id"],
            "pdf_pages": f"{item['start_pdf_page']}-{item['end_pdf_page']}",
            "text": item["digest"],
        }
        for item in items
    ]
    return f"""Classify each report chunk independently.
Select only directly discussed themes from {json.dumps(list(THEMES))}.
Do not infer themes from general development language. Return one classification per chunk,
with the exact chunk_id and a list of present themes. Return JSON only.

CHUNKS:
{json.dumps(compact, ensure_ascii=False)}"""


def report_findings_prompt(chapter_summaries: list[dict]) -> str:
    schema = {
        "report_title": "title",
        "key_results": [
            {"finding": "concise result", "source_pages": [1], "significance": "why it matters"}
        ],
    }
    return f"""Produce 5-6 key results for the full report using only the chapter summaries below.
Prioritize findings supported by quantities or repeated evidence. Do not invent facts.
Keep each finding and significance statement to one concise sentence.
Return only valid JSON matching this structure:
{json.dumps(schema, ensure_ascii=False)}

CHAPTER SUMMARIES:
{json.dumps(chapter_summaries, ensure_ascii=False)}"""


def themes_prompt(chunk: dict) -> str:
    theme_object = {
        theme: {"present": 0, "confidence": 0.0, "evidence": "", "pdf_page": None}
        for theme in THEMES
    }
    schema = {"chunk_id": chunk["chunk_id"], "themes": theme_object}
    return f"""Classify explicit thematic content in a UN development report passage.
Themes: {', '.join(THEMES)}.
For each theme, present must be 1 only when directly discussed and otherwise 0.
Confidence must be between 0 and 1. Evidence must be a short exact phrase when present,
or an empty string when absent. Cite a PDF page in the supplied range when present.
Multiple themes may be present. Return only valid JSON matching this structure:
{json.dumps(schema, ensure_ascii=False)}

{_source_block(chunk['text'], chunk['start_pdf_page'], chunk['end_pdf_page'])}"""


def indicators_prompt(context_chunks: list[dict], table_context: list[dict]) -> str:
    schema = {
        "indicators": [
            {
                "name": "HDI value",
                "value": 0.0,
                "unit": "index",
                "year": 2016,
                "population_group": "total population",
                "pdf_page": 1,
                "evidence": "short source phrase or table row",
            }
        ],
        "not_found": ["HDI rank"],
    }
    context = "\n\n".join(
        _source_block(row["text"], row["start_pdf_page"], row["end_pdf_page"])
        for row in context_chunks
    )
    return f"""Extract development indicators for Timor-Leste from the verified evidence.
Look specifically for: HDI value, HDI rank, life expectancy, expected years of schooling,
mean years of schooling, GNI per capita, and population. Preserve year, unit, and population
group because the report may contain multiple valid values. Never calculate or guess a value.
Put an unavailable requested indicator in not_found. Return only valid JSON matching:
Return at most 7 records: no more than one value for each requested indicator. Put unavailable
indicators in not_found rather than substituting a different metric. Keep evidence below 18 words.
{json.dumps(schema, ensure_ascii=False)}

TEXT EVIDENCE:
{context}

EXTRACTED TABLE EVIDENCE:
{json.dumps(table_context, ensure_ascii=False)}"""


def strengths_challenges_prompt(context_chunks: list[dict]) -> str:
    schema = {
        "strengths": [
            {"item": "concise strength", "explanation": "source-grounded explanation", "source_pages": [1]}
        ],
        "challenges": [
            {"item": "concise challenge", "explanation": "source-grounded explanation", "source_pages": [1]}
        ],
    }
    context = "\n\n".join(
        _source_block(row["text"], row["start_pdf_page"], row["end_pdf_page"])
        for row in context_chunks
    )
    return f"""Identify 5-8 development strengths and 5-8 development challenges for Timor-Leste.
Use only the supplied report evidence. Merge duplicates, distinguish current conditions from
recommendations, and avoid treating a proposed policy as an achieved strength.
Keep each explanation below 25 words.
Return only valid JSON matching this structure:
{json.dumps(schema, ensure_ascii=False)}

{context}"""


def trends_prompt(table_candidates: list[dict]) -> str:
    schema = {
        "series": [
            {
                "series_name": "Adult literacy, ages 15+",
                "unit": "percent",
                "population_group": "ages 15+",
                "pdf_page": 1,
                "points": [{"year": "1996/97", "value": 40.6}],
            }
        ]
    }
    return f"""Convert the extracted report table candidates into demographic or development time series.
Keep only genuine quantities observed at three or more time points. Preserve reported year labels,
units, scenario names, and population groups. Do not interpolate or invent missing values.
Return only valid JSON matching this structure:
{json.dumps(schema, ensure_ascii=False)}

TABLE CANDIDATES:
{json.dumps(table_candidates, ensure_ascii=False)}"""


def evaluation_prompt(
    task_name: str,
    source_evidence: str,
    output: dict,
    required_items: list[str],
) -> str:
    schema = {
        "task": task_name,
        "scores": {
            "factual_alignment": 1,
            "completeness": 1,
            "consistency": 1,
            "format_compliance": 1,
        },
        "unsupported_claims": [],
        "missing_items": [],
        "verdict": "pass or revise",
        "justification": "specific concise explanation",
    }
    return f"""You are an independent evaluator of another local LLM's output.
Compare the output only against the verified source evidence. Do not reward plausible external facts.
Score each criterion from 1 (poor) to 5 (excellent). Be strict about numerical values, years,
population groups, page citations, contradictions, and the task requirements.
List exact unsupported claims and exact missing items; use an empty array when there are none.
Never copy generic wording from the schema example. Verdict must be "revise" if any score is
below 4 or either issue array is non-empty. A score of 4 means the output is reliable with only
minor non-factual limitations; reserve 5 for fully supported and complete work. Treat a correct number
attached to the wrong indicator, year, population group, or statistical definition as unsupported.
Every discrepancy mentioned in the justification must also appear in unsupported_claims or
missing_items. Do not claim something is missing if it is visibly present in the output. Scores,
issue arrays, verdict, and justification must agree with one another. The justification must cite
specific evidence and state clearly when no material factual problem was found.
Return only valid JSON matching this structure:
{json.dumps(schema, ensure_ascii=False)}

TASK: {task_name}
REQUIRED ITEMS: {json.dumps(required_items, ensure_ascii=False)}
VERIFIED SOURCE EVIDENCE:
{source_evidence}

OUTPUT TO EVALUATE:
{json.dumps(output, ensure_ascii=False)}"""


def comparison_prompt(chunk: dict) -> str:
    schema = {
        "summary": "maximum 80 words",
        "themes": ["education"],
        "numerical_facts": [
            {
                "label": "indicator",
                "value": "reported value",
                "unit": "reported unit or not_reported",
                "year": "reported year or null",
                "pdf_page": chunk["start_pdf_page"],
                "evidence": "short exact supporting phrase",
            }
        ],
    }
    return f"""Analyze this UN development report passage using only its contents.
Write a factual summary of at most 80 words, select all applicable themes from
{json.dumps(list(THEMES))}, and extract only explicitly reported numerical facts. Every numerical
fact must include its unit, year (or null), PDF page, and a short supporting phrase. Use
"not_reported" for a genuinely absent unit rather than guessing.
Return valid JSON only, matching this structure:
{json.dumps(schema, ensure_ascii=False)}

{_source_block(chunk['text'], chunk['start_pdf_page'], chunk['end_pdf_page'])}"""
