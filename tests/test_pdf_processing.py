from development_intelligence.pdf_processing import (
    PageRecord,
    clean_page_text,
    create_chunks,
    detect_sections,
)


def page(number: int, text: str) -> PageRecord:
    return PageRecord(number, None, text, len(text))


def test_clean_page_text_repairs_soft_line_breaks():
    raw = "Development Header\nHuman develop-\nment improves.\nNext paragraph."
    cleaned = clean_page_text(raw, {"Development Header"})
    assert "Human development improves." in cleaned
    assert "Development Header" not in cleaned


def test_section_detection_ignores_contents_page():
    pages = [
        page(1, "Contents\nExecutive Summary\nCHAPTER 1"),
        page(2, "Executive Summary\nEvidence"),
        page(3, "CHAPTER 1\nPLANNING THE OPPORTUNITIES FOR YOUTH"),
        page(4, "CHAPTER 2\nYOUTH WELL-BEING"),
    ]
    sections = detect_sections(pages)
    assert [(row.section_id, row.start_pdf_page) for row in sections] == [
        ("executive_summary", 2),
        ("chapter_1", 3),
        ("chapter_2", 4),
    ]


def test_chunks_retain_section_and_page_provenance():
    pages = [page(1, "A" * 120), page(2, "B" * 120), page(3, "C" * 120)]
    sections = detect_sections(
        [page(1, "Executive Summary " + "A" * 100), page(2, "B" * 120), page(3, "C" * 120)]
    )
    chunks = create_chunks(pages, sections, chunk_size=180, overlap=20)
    assert chunks
    assert chunks[0].section_id == "executive_summary"
    assert chunks[0].start_pdf_page <= chunks[0].end_pdf_page


def test_invalid_chunk_configuration_is_rejected():
    try:
        create_chunks([], [], chunk_size=100, overlap=100)
    except ValueError as error:
        assert "chunk_size" in str(error)
    else:
        raise AssertionError("Expected ValueError")

