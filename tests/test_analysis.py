from development_intelligence.analysis import (
    aggregate_theme_results,
    derive_table_time_series,
    enforce_summary_word_limit,
    keyword_theme_baseline,
    theme_evidence_snippet,
    validated_report_findings,
    validated_strengths_challenges,
)


def test_keyword_theme_baseline_counts_each_chunk_once_per_theme():
    chunks = [
        {
            "chunk_id": "c1",
            "section_id": "chapter_1",
            "start_pdf_page": 1,
            "end_pdf_page": 1,
            "text": "Education and school training support employment and jobs.",
        }
    ]
    rows = keyword_theme_baseline(chunks)
    totals = {row["theme"]: row["count"] for row in aggregate_theme_results(rows)}
    assert totals["education"] == 1
    assert totals["employment"] == 1
    assert totals["health"] == 0


def test_theme_evidence_snippet_returns_source_sentence():
    text = "The economy remains constrained. Education and training support youth employment."
    evidence = theme_evidence_snippet(text, "education")
    assert evidence == "Education and training support youth employment."


def test_time_series_reconstruction_from_split_pdf_tables():
    tables = [
        {
            "table_id": "p010_t01",
            "pdf_page": 10,
            "rows": [["Indicator", "2000", "2010", "2020"]],
        },
        {
            "table_id": "p010_t02",
            "pdf_page": 10,
            "rows": [["Adult literacy, %", "40.0", "50.5", "67.2"]],
        },
    ]
    series = derive_table_time_series(tables)
    assert len(series) == 1
    assert series[0]["series_name"] == "Adult literacy, %"
    assert series[0]["points"][-1] == {"year": "2020", "value": 67.2}


def test_summary_word_limit_is_enforced_and_audited():
    payload = {"summary": " ".join(f"word{index}" for index in range(120))}
    result = enforce_summary_word_limit(payload, maximum=99)
    assert len(result["summary"].split()) <= 99
    assert result["word_limit_applied"] is True
    assert "summary_original" in result


def test_validated_report_findings_corrects_and_deduplicates_overview():
    raw = {"report_title": "Test report", "key_results": [{"finding": "duplicate"}]}
    result = validated_report_findings(raw)

    assert result["report_title"] == "Test report"
    assert len(result["key_results"]) == 5
    assert [row["source_pages"] for row in result["key_results"]] == [
        [17], [18], [19], [20], [62]
    ]
    assert "separately, 46%" in result["key_results"][1]["finding"]
    assert "by 2050" not in " ".join(
        row["finding"] for row in result["key_results"]
    )


def test_validated_strengths_contains_only_documented_achievements():
    result = validated_strengths_challenges({"source_model": "test-model"})

    assert len(result["strengths"]) == 5
    assert len(result["challenges"]) == 5
    assert {row["classification"] for row in result["strengths"]} == {"Achievement"}
    assert all(row["evidence"] and row["source_pages"] for row in result["strengths"])
    assert all(row["evidence"] and row["source_pages"] for row in result["challenges"])
    joined = " ".join(row["item"] for row in result["strengths"])
    assert "investment target" not in joined.lower()
    assert "parliamentary representation" in joined.lower()
