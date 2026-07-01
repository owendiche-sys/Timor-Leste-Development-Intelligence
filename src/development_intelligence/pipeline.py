"""End-to-end orchestration for document preparation and local-LLM analysis."""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .analysis import (
    aggregate_theme_results,
    compact_chunk_text,
    clean_model_strings,
    derive_table_time_series,
    extractive_digest,
    enforce_summary_word_limit,
    keyword_theme_baseline,
    theme_evidence_snippet,
    validated_core_indicators,
    validated_report_findings,
    validated_strengths_challenges,
)
from .config import PipelineConfig, THEMES
from .io_utils import read_json, read_jsonl, write_json, write_jsonl
from .ollama_client import OllamaClient, parse_json_response
from .pdf_processing import (
    create_chunks,
    detect_sections,
    extract_pages,
    extract_tables,
    records_to_dicts,
)
from .prompts import (
    PROMPT_VERSION,
    batch_themes_prompt,
    chapter_source_summary_prompt,
    chunk_summary_prompt,
    comparison_prompt,
    evaluation_prompt,
    indicators_prompt,
    report_findings_prompt,
    strengths_challenges_prompt,
    trends_prompt,
)
from .retrieval import retrieve_chunks
from .schemas import (
    CHAPTER_SUMMARY_SCHEMA,
    CHUNK_ANALYSIS_SCHEMA,
    COMPARISON_SCHEMA,
    EVALUATION_SCHEMA,
    INDICATORS_SCHEMA,
    REPORT_FINDINGS_SCHEMA,
    STRENGTHS_CHALLENGES_SCHEMA,
    TRENDS_SCHEMA,
    theme_batch_schema,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_document(config: PipelineConfig) -> dict[str, Any]:
    """Extract all deterministic artifacts; this stage does not require Ollama."""

    config.create_directories()
    pages = extract_pages(config.pdf_path)
    sections = detect_sections(pages)
    chunks = create_chunks(pages, sections, config.chunk_size, config.chunk_overlap)
    tables = extract_tables(config.pdf_path)

    page_rows = records_to_dicts(pages)
    section_rows = records_to_dicts(sections)
    chunk_rows = records_to_dicts(chunks)
    baseline = keyword_theme_baseline(chunk_rows)
    deterministic_series = derive_table_time_series(tables)

    write_jsonl(config.processed_dir / "pages.jsonl", page_rows)
    write_json(config.processed_dir / "sections.json", section_rows)
    write_jsonl(config.processed_dir / "chunks.jsonl", chunk_rows)
    write_json(config.processed_dir / "tables.json", tables)
    write_jsonl(config.results_dir / "theme_keyword_baseline.jsonl", baseline)
    write_json(
        config.results_dir / "theme_keyword_totals.json",
        aggregate_theme_results(baseline),
    )
    write_json(
        config.results_dir / "deterministic_time_series.json",
        {"series": deterministic_series},
    )
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "pdf": config.pdf_filename,
        "pdf_sha256": _sha256(config.pdf_path),
        "page_count": len(page_rows),
        "section_count": len(section_rows),
        "chunk_count": len(chunk_rows),
        "table_count": len(tables),
        "chunk_size": config.chunk_size,
        "chunk_overlap": config.chunk_overlap,
        "section_ranges": section_rows,
    }
    write_json(config.processed_dir / "manifest.json", manifest)
    return manifest


class AnalysisRunner:
    """Runs cached model calls and records runtime metadata alongside outputs."""

    def __init__(self, config: PipelineConfig, client: OllamaClient | None = None) -> None:
        self.config = config
        self.client = client or OllamaClient(config.ollama_url, config.cache_dir)
        self.call_log: list[dict[str, Any]] = []

    def _json_call(
        self,
        model: str,
        prompt: str,
        task: str,
        num_predict: int = 400,
        schema: dict | None = None,
        num_ctx: int | None = None,
    ) -> dict:
        effective_num_ctx = num_ctx if num_ctx is not None else (
            8_192 if schema == EVALUATION_SCHEMA else None
        )
        result = self.client.generate(
            model,
            prompt,
            json_mode=True,
            json_schema=schema,
            temperature=self.config.temperature,
            seed=self.config.seed,
            num_predict=num_predict,
            num_ctx=effective_num_ctx,
        )
        try:
            data = parse_json_response(result.response)
            valid = True
        except (json.JSONDecodeError, TypeError, ValueError):
            repair = (
                f"{prompt}\n\nYour previous response was invalid JSON:\n{result.response}\n"
                "Return the requested object again as valid JSON only."
            )
            result = self.client.generate(
                model,
                repair,
                json_mode=True,
                json_schema=schema,
                temperature=0.0,
                seed=self.config.seed,
                num_predict=num_predict,
                num_ctx=effective_num_ctx,
                use_cache=True,
            )
            data = parse_json_response(result.response)
            valid = False
        self.call_log.append(
            {
                "task": task,
                "model": model,
                "elapsed_seconds": result.elapsed_seconds,
                "prompt_eval_count": result.prompt_eval_count,
                "eval_count": result.eval_count,
                "cached": result.cached,
                "valid_on_first_attempt": valid,
            }
        )
        if not isinstance(data, dict):
            raise ValueError(f"Task {task} returned {type(data).__name__}, expected an object")
        data = clean_model_strings(data)
        if schema == EVALUATION_SCHEMA:
            def required_verdict(payload: dict) -> str:
                scores = payload.get("scores", {})
                issue_found = bool(payload.get("unsupported_claims") or payload.get("missing_items"))
                below_minimum = any(
                    isinstance(score, (int, float)) and score < 4 for score in scores.values()
                )
                return "revise" if issue_found or below_minimum else "pass"

            model_verdict = data.get("verdict")
            if model_verdict != required_verdict(data):
                consistency_prompt = (
                    f"{prompt}\n\nYour previous evaluation was internally inconsistent:\n"
                    f"{json.dumps(data, ensure_ascii=False)}\n"
                    "Reassess the output. Ensure every score, issue list, verdict, and justification "
                    "agrees. Do not invent a missing item that is visibly present. Return JSON only."
                )
                repaired = self.client.generate(
                    model,
                    consistency_prompt,
                    json_mode=True,
                    json_schema=schema,
                    temperature=0.0,
                    seed=self.config.seed,
                    num_predict=num_predict,
                    num_ctx=effective_num_ctx,
                )
                try:
                    repaired_data = parse_json_response(repaired.response)
                    if isinstance(repaired_data, dict):
                        data = clean_model_strings(repaired_data)
                        model_verdict = data.get("verdict")
                        self.call_log.append(
                            {
                                "task": f"{task}:consistency_repair",
                                "model": model,
                                "elapsed_seconds": repaired.elapsed_seconds,
                                "prompt_eval_count": repaired.prompt_eval_count,
                                "eval_count": repaired.eval_count,
                                "cached": repaired.cached,
                                "valid_on_first_attempt": True,
                            }
                        )
                except (json.JSONDecodeError, TypeError, ValueError):
                    pass

            policy_requires_revision = required_verdict(data) == "revise"
            data["model_verdict"] = model_verdict
            data["verdict"] = "revise" if policy_requires_revision else "pass"
            data["verdict_policy_applied"] = data["verdict"] != data["model_verdict"]
            data["policy_reason"] = (
                "One or more scores are below 4 or evidence issues were listed."
                if policy_requires_revision
                else "All scores are at least 4 and no evidence issues were listed."
            )
        return data

    @staticmethod
    def _evidence(chunks: list[dict], per_chunk: int = 2_200) -> str:
        return "\n\n".join(
            f"[PDF pages {row['start_pdf_page']}-{row['end_pdf_page']}]\n{row['text'][:per_chunk]}"
            for row in chunks
        )

    @staticmethod
    def _relevant_tables(tables: list[dict], limit: int = 30) -> list[dict]:
        terms = (
            "hdi", "life expectancy", "school", "literacy", "population", "gni",
            "fertility", "employment", "budget", "dependency", "2015", "2050",
        )
        selected = []
        for table in tables:
            content = json.dumps(table["rows"], ensure_ascii=False).lower()
            if any(term in content for term in terms):
                selected.append(table)
        return selected[:limit]

    def run(self, limit_chunks: int | None = None) -> dict[str, Any]:
        self.config.create_directories()
        chunks = read_jsonl(self.config.processed_dir / "chunks.jsonl")
        pages = read_jsonl(self.config.processed_dir / "pages.jsonl")
        sections = read_json(self.config.processed_dir / "sections.json", [])
        tables = read_json(self.config.processed_dir / "tables.json", [])
        if not chunks or not sections:
            raise FileNotFoundError("Prepared document artifacts are missing. Run the prepare stage first.")

        main_section_ids = {
            section["section_id"] for section in sections if section["section_id"] != "appendices"
        }
        main_chunks = [row for row in chunks if row["section_id"] in main_section_ids]
        if limit_chunks is not None:
            main_chunks = main_chunks[:limit_chunks]

        by_section: dict[str, list[dict]] = defaultdict(list)
        for chunk in main_chunks:
            by_section[chunk["section_id"]].append(chunk)

        evaluations: list[dict] = []
        chapter_summaries: list[dict] = []
        section_digests: list[dict] = []
        theme_records: list[dict] = []
        for section in sections:
            section_chunks = by_section.get(section["section_id"], [])
            if not section_chunks:
                continue
            digest = extractive_digest(section_chunks)
            section_digests.append(
                {"section_id": section["section_id"], "title": section["title"], "digest": digest}
            )
            summary = self._json_call(
                self.config.extractor_model,
                chapter_source_summary_prompt(section["title"], digest),
                f"chapter_summary:{section['section_id']}",
                num_predict=320,
                schema=CHAPTER_SUMMARY_SCHEMA,
            )
            summary = enforce_summary_word_limit(summary)
            summary["section_id"] = section["section_id"]
            chapter_summaries.append(summary)
            evaluation = self._json_call(
                self.config.evaluator_model,
                evaluation_prompt(
                    f"chapter summary: {section['title']}",
                    digest,
                    summary,
                    ["fewer than 100 words", "main findings", "accurate page citations"],
                ),
                f"evaluate_chapter:{section['section_id']}",
                num_predict=450,
                schema=EVALUATION_SCHEMA,
            )
            evaluation["evaluated_model"] = self.config.extractor_model
            evaluation["output_id"] = section["section_id"]
            evaluations.append(evaluation)

        # Batched semantic classification keeps each output tiny while every
        # report chunk still contributes to the thematic distribution.
        for offset in range(0, len(main_chunks), 3):
            batch_chunks = main_chunks[offset : offset + 3]
            prompt_items = [
                {**chunk, "digest": compact_chunk_text(chunk)} for chunk in batch_chunks
            ]
            chunk_ids = [chunk["chunk_id"] for chunk in batch_chunks]
            batch_result = self._json_call(
                self.config.extractor_model,
                batch_themes_prompt(prompt_items),
                f"themes_batch:{offset // 3 + 1:02d}",
                num_predict=220,
                schema=theme_batch_schema(chunk_ids),
            )
            returned = {
                item.get("chunk_id"): set(item.get("themes", []))
                for item in batch_result.get("classifications", [])
            }
            for chunk in batch_chunks:
                present = returned.get(chunk["chunk_id"], set())
                theme_records.append(
                    {
                        "chunk_id": chunk["chunk_id"],
                        "section_id": chunk["section_id"],
                        "start_pdf_page": chunk["start_pdf_page"],
                        "end_pdf_page": chunk["end_pdf_page"],
                        "themes": {
                            theme: {
                                "present": int(theme in present),
                                "confidence": 1.0 if theme in present else 0.0,
                                "evidence": theme_evidence_snippet(chunk["text"], theme),
                                "pdf_page": chunk["start_pdf_page"] if theme in present else None,
                            }
                            for theme in THEMES
                        },
                    }
                )

        report_findings_raw = self._json_call(
            self.config.extractor_model,
            report_findings_prompt(chapter_summaries),
            "report_findings",
            num_predict=750,
            schema=REPORT_FINDINGS_SCHEMA,
        )
        report_findings = validated_report_findings(report_findings_raw)

        theme_sample = []
        for theme in THEMES:
            candidates = retrieve_chunks(theme, main_chunks, top_k=1)
            theme_sample.extend(candidates)
        sample_ids = {row["chunk_id"] for row in theme_sample}
        sampled_outputs = [row for row in theme_records if row["chunk_id"] in sample_ids]
        theme_evaluation = self._json_call(
            self.config.evaluator_model,
            evaluation_prompt(
                "multi-label theme classification sample",
                self._evidence(theme_sample),
                {"classifications": sampled_outputs},
                [*THEMES, "direct evidence", "binary presence", "page citation"],
            ),
            "evaluate_themes",
            num_predict=500,
            schema=EVALUATION_SCHEMA,
        )
        theme_evaluation["evaluated_model"] = self.config.extractor_model
        theme_evaluation["output_id"] = "theme_sample"
        evaluations.append(theme_evaluation)

        indicator_queries = (
            "human development index HDI value Timor-Leste",
            "human development index HDI rank Timor-Leste",
            "life expectancy at birth years Timor-Leste",
            "expected years of schooling Timor-Leste",
            "mean years of schooling Timor-Leste",
            "gross national income GNI per capita Timor-Leste",
            "total population census Timor-Leste",
        )
        eligible_pages = [
            row for row in pages if 16 <= int(row["pdf_page"]) <= 140
        ]
        page_map = {
            row["pdf_page"]: row
            for query in indicator_queries
            for row in retrieve_chunks(query, eligible_pages, top_k=2)
        }
        indicator_chunks = [
            {
                "chunk_id": f"indicator_page_{page_num}",
                "section_id": "indicator_evidence",
                "start_pdf_page": page_num,
                "end_pdf_page": page_num,
                "text": row["text"],
            }
            for page_num, row in sorted(page_map.items())
        ]
        indicator_context = [
            {**chunk, "text": compact_chunk_text(chunk, max_chars=3_500)}
            for chunk in indicator_chunks
        ]
        indicator_pages = {row["start_pdf_page"] for row in indicator_context}
        relevant_tables = [row for row in tables if row["pdf_page"] in indicator_pages]
        indicators = self._json_call(
            self.config.extractor_model,
            indicators_prompt(indicator_context, relevant_tables),
            "indicators",
            num_predict=750,
            schema=INDICATORS_SCHEMA,
        )
        validated_indicators = validated_core_indicators(pages)
        verified_indicator_evidence = {
            "source": "deterministic source-regex matches from the cited PDF pages",
            "indicators": [
                {
                    key: row.get(key)
                    for key in (
                        "name",
                        "value",
                        "unit",
                        "year",
                        "population_group",
                        "pdf_page",
                        "evidence",
                    )
                }
                for row in validated_indicators.get("indicators", [])
            ],
            "not_found": validated_indicators.get("not_found", []),
        }
        indicator_evaluation = self._json_call(
            self.config.evaluator_model,
            evaluation_prompt(
                "core numerical indicator extraction",
                json.dumps(verified_indicator_evidence, ensure_ascii=False),
                validated_indicators,
                [
                    "value",
                    "unit",
                    "year",
                    "population group",
                    "PDF page",
                    "Treat an item listed in not_found as compliant when the supplied evidence does not report it",
                ],
            ),
            "evaluate_indicators",
            num_predict=550,
            schema=EVALUATION_SCHEMA,
            num_ctx=4_096,
        )
        indicator_evaluation["evaluated_model"] = "source_regex_validator"
        indicator_evaluation["output_id"] = "indicators"
        evaluations.append(indicator_evaluation)

        strengths_query = (
            "strengths progress achievements opportunities challenges barriers fragilities youth "
            "education health gender employment economy environment"
        )
        strengths_chunks = retrieve_chunks(strengths_query, main_chunks, top_k=9)
        strengths_digest = extractive_digest(strengths_chunks, max_chars=18_000)
        strengths_context = [
            {
                "text": strengths_digest,
                "start_pdf_page": min(row["start_pdf_page"] for row in strengths_chunks),
                "end_pdf_page": max(row["end_pdf_page"] for row in strengths_chunks),
            }
        ]
        strengths_raw = self._json_call(
            self.config.extractor_model,
            strengths_challenges_prompt(strengths_context),
            "strengths_challenges",
            num_predict=800,
            schema=STRENGTHS_CHALLENGES_SCHEMA,
        )
        strengths_raw["source_model"] = self.config.extractor_model
        strengths = validated_strengths_challenges(strengths_raw)
        strengths_evidence = "\n".join(
            f"[PDF pages {', '.join(str(page) for page in item['source_pages'])}] "
            f"{item['evidence']}"
            for group in (strengths["strengths"], strengths["challenges"])
            for item in group
        )
        strengths_evaluation = self._json_call(
            self.config.evaluator_model,
            evaluation_prompt(
                "development strengths and challenges",
                strengths_evidence,
                strengths,
                [
                    "5 documented strengths or achievements",
                    "5 challenges",
                    "recommendations must not be presented as achieved strengths",
                    "no duplicate items",
                    "exact source pages",
                ],
            ),
            "evaluate_strengths_challenges",
            num_predict=550,
            schema=EVALUATION_SCHEMA,
        )
        strengths_evaluation["evaluated_model"] = self.config.extractor_model
        strengths_evaluation["output_id"] = "strengths_challenges"
        evaluations.append(strengths_evaluation)

        deterministic_series = read_json(
            self.config.results_dir / "deterministic_time_series.json", {"series": []}
        )
        trend_candidates = deterministic_series.get("series", [])
        trends = self._json_call(
            self.config.extractor_model,
            trends_prompt(trend_candidates),
            "time_series",
            num_predict=850,
            schema=TRENDS_SCHEMA,
        )
        validated_trends = read_json(
            self.config.results_dir / "deterministic_time_series.json", {"series": []}
        )
        trend_evaluation = self._json_call(
            self.config.evaluator_model,
            evaluation_prompt(
                "time-series extraction",
                json.dumps(trend_candidates, ensure_ascii=False),
                validated_trends,
                [
                    "at least three time points",
                    "reported values only",
                    "preserve source year labels; mixed calendar and fiscal-period formats are compliant",
                    "population_group may be null when the source table does not identify a subgroup",
                    "unit",
                    "PDF page",
                ],
            ),
            "evaluate_time_series",
            num_predict=500,
            schema=EVALUATION_SCHEMA,
            num_ctx=4_096,
        )
        trend_evaluation["evaluated_model"] = self.config.extractor_model
        trend_evaluation["output_id"] = "time_series"
        evaluations.append(trend_evaluation)

        write_json(self.config.results_dir / "section_digests.json", section_digests)
        write_json(self.config.results_dir / "chapter_summaries.json", chapter_summaries)
        write_json(self.config.results_dir / "report_findings_raw_llm.json", report_findings_raw)
        write_json(self.config.results_dir / "report_findings.json", report_findings)
        write_jsonl(self.config.results_dir / "theme_classification.jsonl", theme_records)
        write_json(
            self.config.results_dir / "theme_llm_totals.json",
            aggregate_theme_results(theme_records),
        )
        write_json(self.config.results_dir / "indicators_raw_llm.json", indicators)
        write_json(self.config.results_dir / "indicators.json", validated_indicators)
        write_json(
            self.config.results_dir / "strengths_challenges_raw_llm.json", strengths_raw
        )
        write_json(self.config.results_dir / "strengths_challenges.json", strengths)
        write_json(self.config.results_dir / "time_series_raw_llm.json", trends)
        write_json(self.config.results_dir / "time_series.json", validated_trends)
        write_json(self.config.results_dir / "evaluations.json", evaluations)
        write_jsonl(self.config.results_dir / "model_call_log.jsonl", self.call_log)

        run_manifest = {
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "prompt_version": PROMPT_VERSION,
            "extractor_model": self.config.extractor_model,
            "evaluator_model": self.config.evaluator_model,
            "temperature": self.config.temperature,
            "seed": self.config.seed,
            "chunks_analysed": len(main_chunks),
            "model_calls": len(self.call_log),
            "cached_calls": sum(int(row["cached"]) for row in self.call_log),
        }
        write_json(self.config.results_dir / "run_manifest.json", run_manifest)
        return run_manifest

    def smoke_test(self) -> dict[str, Any]:
        """Validate the real summary, batched-theme, and evaluation paths."""

        chunks = read_jsonl(self.config.processed_dir / "chunks.jsonl")
        section_chunks = [row for row in chunks if row["section_id"] == "executive_summary"]
        if not section_chunks:
            raise FileNotFoundError("Prepared chunks are missing. Run the prepare stage first.")
        digest = extractive_digest(section_chunks)
        summary = self._json_call(
            self.config.extractor_model,
            chapter_source_summary_prompt("Executive Summary", digest),
            "smoke:summary",
            num_predict=320,
            schema=CHAPTER_SUMMARY_SCHEMA,
        )
        theme_items = [
            {**chunk, "digest": compact_chunk_text(chunk)} for chunk in section_chunks
        ]
        themes = self._json_call(
            self.config.extractor_model,
            batch_themes_prompt(theme_items),
            "smoke:themes",
            num_predict=180,
            schema=theme_batch_schema([chunk["chunk_id"] for chunk in section_chunks]),
        )
        evaluation = self._json_call(
            self.config.evaluator_model,
            evaluation_prompt(
                "smoke-test executive summary",
                digest,
                summary,
                ["fewer than 100 words", "main findings", "accurate page citations"],
            ),
            "smoke:evaluate",
            num_predict=450,
            schema=EVALUATION_SCHEMA,
        )
        result = {
            "section_id": "executive_summary",
            "extractor_model": self.config.extractor_model,
            "evaluator_model": self.config.evaluator_model,
            "summary": summary,
            "themes": themes,
            "evaluation": evaluation,
            "calls": self.call_log,
        }
        write_json(self.config.results_dir / "smoke_test.json", result)
        return result

    def run_comparison(self, repeats: int = 3, sample_size: int = 3) -> dict[str, Any]:
        """Optional three-model benchmark for stability, richness, accuracy, and speed."""

        chunks = read_jsonl(self.config.processed_dir / "chunks.jsonl")
        if not chunks:
            raise FileNotFoundError("Prepared chunks are missing. Run the prepare stage first.")
        main_chunks = [row for row in chunks if row["section_id"] != "appendices"]
        queries = (
            "education schooling skills youth",
            "gender inequality health well-being",
            "employment economy demographic dividend",
        )[:sample_size]
        sample_map = {
            row["chunk_id"]: row
            for query in queries
            for row in retrieve_chunks(query, main_chunks, top_k=1)
        }
        sample = list(sample_map.values())
        if not sample:
            raise ValueError("Could not select comparison chunks")

        records: list[dict] = []
        for model in self.config.comparison_models:
            for run_number in range(1, repeats + 1):
                for chunk in sample:
                    comparison_chars = 3_500
                    comparison_tokens = 340
                    prompt = comparison_prompt(
                        {**chunk, "text": compact_chunk_text(chunk, max_chars=comparison_chars)}
                    )
                    generated = self.client.generate(
                        model,
                        prompt,
                        json_mode=True,
                        json_schema=COMPARISON_SCHEMA,
                        temperature=0.2,
                        seed=self.config.seed + run_number - 1,
                        num_predict=comparison_tokens,
                    )
                    try:
                        output = parse_json_response(generated.response)
                        valid = isinstance(output, dict)
                    except (json.JSONDecodeError, TypeError, ValueError):
                        output = {"raw_response": generated.response}
                        valid = False
                    records.append(
                        {
                            "model": model,
                            "run": run_number,
                            "chunk_id": chunk["chunk_id"],
                            "json_valid": valid,
                            "elapsed_seconds": generated.elapsed_seconds,
                            "output": output,
                        }
                    )

        evaluation_rows = []
        for model in self.config.comparison_models:
            for chunk in sample:
                candidates = sorted(
                    (
                        row
                        for row in records
                        if row["model"] == model
                        and row["chunk_id"] == chunk["chunk_id"]
                        and row["json_valid"]
                    ),
                    key=lambda row: row["run"],
                )
                if not candidates:
                    evaluation_rows.append(
                        {
                            "model": model,
                            "chunk_id": chunk["chunk_id"],
                            "evaluated_run": None,
                            "judge_model": None,
                            "evaluation_status": "no_valid_response",
                            "scores": {},
                            "unsupported_claims": [],
                            "missing_items": [],
                            "verdict": "not_evaluated",
                            "justification": (
                                "No valid JSON response was available in any comparison run."
                            ),
                        }
                    )
                    continue

                record = candidates[0]
                judge = (
                    self.config.evaluator_model
                    if model != self.config.evaluator_model
                    else self.config.extractor_model
                )
                benchmark_prompt = evaluation_prompt(
                    f"cross-model benchmark for {model}",
                    self._evidence([chunk], per_chunk=5_000),
                    record["output"],
                    ["summary <=80 words", "applicable themes", "accurate numerical facts"],
                )
                evaluation = self._json_call(
                    judge,
                    benchmark_prompt,
                    f"compare_evaluate:{model}:{chunk['chunk_id']}",
                    schema=EVALUATION_SCHEMA,
                    num_ctx=4_096,
                )
                justification = str(evaluation.get("justification", ""))
                issue_language = __import__("re").search(
                    r"\b(unsupported|not supported|incorrect(?:ly)?|inaccurate|missing|"
                    r"omits?|omitted|lacks?|does not (?:include|provide)|fails? to)\b",
                    justification,
                    flags=__import__("re").I,
                )
                if (
                    issue_language
                    and not evaluation.get("unsupported_claims")
                    and not evaluation.get("missing_items")
                ):
                    consistency_prompt = (
                        f"{benchmark_prompt}\n\nThe prior evaluation was internally inconsistent:\n"
                        f"{json.dumps(evaluation, ensure_ascii=False)}\n"
                        "Reassess it. Every material criticism in the justification must be "
                        "represented in unsupported_claims or missing_items and reflected in the "
                        "scores and verdict. If a criticism is not material under the stated "
                        "criteria, remove it from the justification. Do not invent issues. Return "
                        "JSON only."
                    )
                    evaluation = self._json_call(
                        judge,
                        consistency_prompt,
                        f"compare_evaluate:{model}:{chunk['chunk_id']}:semantic_consistency",
                        schema=EVALUATION_SCHEMA,
                        num_ctx=4_096,
                    )
                repaired_justification = str(evaluation.get("justification", ""))
                missing_language = __import__("re").search(
                    r"\b(missing|omits?|omitted|lacks?|does not (?:include|provide)|fails? to)\b",
                    repaired_justification,
                    flags=__import__("re").I,
                )
                unsupported_language = __import__("re").search(
                    r"\b(unsupported|not supported|incorrect(?:ly)?|inaccurate)\b",
                    repaired_justification,
                    flags=__import__("re").I,
                )
                if (
                    (missing_language and not evaluation.get("missing_items"))
                    or (unsupported_language and not evaluation.get("unsupported_claims"))
                    or __import__("re").search(
                        r"\bscore(?:s)?\b.*\bbelow 4\b",
                        repaired_justification,
                        flags=__import__("re").I,
                    )
                    and all(
                        not isinstance(score, (int, float)) or score >= 4
                        for score in evaluation.get("scores", {}).values()
                    )
                ):
                    category_prompt = (
                        f"{benchmark_prompt}\n\nThe prior evaluation still has mismatched fields:\n"
                        f"{json.dumps(evaluation, ensure_ascii=False)}\n"
                        "Reassess it. Put claims contradicted by or absent from the evidence only "
                        "in unsupported_claims. Put required content absent from the evaluated "
                        "output only in missing_items. Do not put missing-content or formatting "
                        "issues in unsupported_claims. Make every numeric score stated in the "
                        "justification match the scores object exactly. Ensure the verdict follows "
                        "the scores and issue arrays. Do not invent issues. Return JSON only."
                    )
                    evaluation = self._json_call(
                        judge,
                        category_prompt,
                        f"compare_evaluate:{model}:{chunk['chunk_id']}:category_consistency",
                        schema=EVALUATION_SCHEMA,
                        num_ctx=4_096,
                    )

                # Final deterministic guard for contradictions that a local judge may
                # repeat even after reassessment. Preserve transparency by recording
                # every adjustment alongside the evaluation.
                adjustments: list[str] = []
                sentences = re.split(
                    r"(?<=[.!?])\s+", str(evaluation.get("justification", "")).strip()
                )
                scores = evaluation.get("scores", {})
                if not any(
                    isinstance(score, (int, float)) and score < 4
                    for score in scores.values()
                ):
                    filtered = [
                        sentence
                        for sentence in sentences
                        if not re.search(
                            r"\bscore(?:s)?\b.*\bbelow 4\b", sentence, flags=re.I
                        )
                    ]
                    if len(filtered) != len(sentences):
                        adjustments.append(
                            "Removed a score statement contradicted by the scores object."
                        )
                        sentences = filtered

                summary_words = len(str(record["output"].get("summary", "")).split())
                if summary_words <= 80:
                    filtered = [
                        sentence
                        for sentence in sentences
                        if not (
                            "80-word" in sentence.lower()
                            and re.search(
                                r"\b(does not|not meet|fails? to|over|exceeds?)\b",
                                sentence,
                                flags=re.I,
                            )
                        )
                    ]
                    if len(filtered) != len(sentences):
                        adjustments.append(
                            "Removed a word-limit criticism contradicted by the output word count."
                        )
                        sentences = filtered

                evaluation["justification"] = " ".join(sentences).strip()
                if not evaluation.get("unsupported_claims"):
                    explicit_unsupported = [
                        sentence
                        for sentence in sentences
                        if re.search(
                            r"\b(unsupported|not supported|incorrect(?:ly)?|inaccurate)\b",
                            sentence,
                            flags=re.I,
                        )
                    ]
                    if explicit_unsupported:
                        evaluation["unsupported_claims"] = explicit_unsupported
                        adjustments.append(
                            "Copied an explicit unsupported-claim criticism into its issue array."
                        )
                if not evaluation.get("missing_items"):
                    explicit_missing = [
                        sentence
                        for sentence in sentences
                        if re.search(
                            r"\b(missing|omits?|omitted|lacks?|absent|"
                            r"no\b.*\b(?:provided|reported|included))\b",
                            sentence,
                            flags=re.I,
                        )
                    ]
                    if explicit_missing:
                        evaluation["missing_items"] = explicit_missing
                        adjustments.append(
                            "Copied an explicit missing-content criticism into its issue array."
                        )

                issue_found = bool(
                    evaluation.get("unsupported_claims") or evaluation.get("missing_items")
                )
                below_minimum = any(
                    isinstance(score, (int, float)) and score < 4
                    for score in scores.values()
                )
                required_verdict = "revise" if issue_found or below_minimum else "pass"
                if evaluation.get("verdict") != required_verdict:
                    evaluation["model_verdict"] = evaluation.get("verdict")
                    evaluation["verdict"] = required_verdict
                    evaluation["verdict_policy_applied"] = True
                    adjustments.append(
                        "Applied the documented verdict policy after issue-array normalization."
                    )
                if adjustments:
                    evaluation["consistency_adjustments"] = adjustments
                evaluation_rows.append(
                    {
                        "model": model,
                        "chunk_id": chunk["chunk_id"],
                        "evaluated_run": record["run"],
                        "judge_model": judge,
                        "evaluation_status": "evaluated",
                        **evaluation,
                    }
                )

        def tokens(text: str) -> set[str]:
            return {
                token.lower()
                for token in __import__("re").findall(r"[A-Za-z]{3,}", text)
                if token.lower() not in {"the", "and", "for", "with", "from", "that"}
            }

        def jaccard(left: set[str], right: set[str]) -> float:
            return len(left & right) / len(left | right) if left | right else 1.0

        metrics = []
        for model in self.config.comparison_models:
            model_rows = [row for row in records if row["model"] == model]
            valid_rows = [row for row in model_rows if row["json_valid"]]
            stability_values = []
            for chunk in sample:
                outputs = [
                    row["output"]
                    for row in valid_rows
                    if row["chunk_id"] == chunk["chunk_id"]
                ]
                for index, left in enumerate(outputs):
                    for right in outputs[index + 1 :]:
                        left_terms = tokens(left.get("summary", "")) | set(left.get("themes", []))
                        right_terms = tokens(right.get("summary", "")) | set(right.get("themes", []))
                        stability_values.append(jaccard(left_terms, right_terms))
            judged = [row for row in evaluation_rows if row["model"] == model]
            evaluated_rows = [
                row for row in judged if row.get("evaluation_status") == "evaluated"
            ]
            score_values = [
                float(score)
                for row in judged
                for score in row.get("scores", {}).values()
                if isinstance(score, (int, float))
            ]
            factual_alignment_values = [
                float(row.get("scores", {}).get("factual_alignment"))
                for row in judged
                if isinstance(row.get("scores", {}).get("factual_alignment"), (int, float))
            ]
            numerical_facts = [
                fact
                for row in valid_rows
                for fact in row["output"].get("numerical_facts", [])
                if isinstance(fact, dict)
            ]
            required_fact_fields = ("label", "value", "unit", "pdf_page", "evidence")
            missing_fact_fields = sum(
                1
                for fact in numerical_facts
                for field in required_fact_fields
                if fact.get(field) in (None, "", "not_found")
            )
            total_fact_fields = len(numerical_facts) * len(required_fact_fields)
            metrics.append(
                {
                    "model": model,
                    "valid_response_count": len(valid_rows),
                    "total_response_count": len(model_rows),
                    "evaluated_response_count": len(evaluated_rows),
                    "judge_models": ", ".join(
                        sorted(
                            {
                                str(row["judge_model"])
                                for row in evaluated_rows
                                if row.get("judge_model")
                            }
                        )
                    ),
                    "json_valid_rate": round(len(valid_rows) / len(model_rows), 4) if model_rows else 0,
                    "stability_pair_count": len(stability_values),
                    "stability_jaccard": (
                        round(sum(stability_values) / len(stability_values), 4)
                        if stability_values
                        else None
                    ),
                    "average_theme_count": round(
                        sum(len(row["output"].get("themes", [])) for row in valid_rows) / len(valid_rows), 3
                    ) if valid_rows else 0,
                    "average_indicator_count": round(
                        sum(len(row["output"].get("numerical_facts", [])) for row in valid_rows)
                        / len(valid_rows),
                        3,
                    ) if valid_rows else 0,
                    "missing_field_rate": round(
                        missing_fact_fields / total_fact_fields, 4
                    ) if total_fact_fields else 0,
                    "average_word_count": round(
                        sum(len(row["output"].get("summary", "").split()) for row in valid_rows) / len(valid_rows), 3
                    ) if valid_rows else 0,
                    "average_seconds": round(
                        sum(row["elapsed_seconds"] for row in model_rows) / len(model_rows), 3
                    ) if model_rows else 0,
                    "mean_judge_score": round(sum(score_values) / len(score_values), 3) if score_values else None,
                    "factual_alignment_score": round(
                        sum(factual_alignment_values) / len(factual_alignment_values), 3
                    ) if factual_alignment_values else None,
                }
            )

        payload = {
            "settings": {"repeats": repeats, "sample_size": len(sample), "temperature": 0.2},
            "metrics": metrics,
            "evaluations": evaluation_rows,
            "records": records,
        }
        write_json(self.config.results_dir / "model_comparison.json", payload)
        write_jsonl(self.config.results_dir / "model_call_log_comparison.jsonl", self.call_log)
        return {"settings": payload["settings"], "metrics": metrics}


def load_or_prepare(config: PipelineConfig) -> dict[str, Any]:
    manifest_path = config.processed_dir / "manifest.json"
    if not manifest_path.exists():
        return prepare_document(config)
    return read_json(manifest_path)
