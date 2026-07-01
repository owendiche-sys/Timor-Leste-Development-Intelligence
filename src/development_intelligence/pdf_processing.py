"""PDF extraction, cleaning, section detection, chunking, and table capture."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from pypdf import PdfReader


@dataclass(frozen=True)
class PageRecord:
    pdf_page: int
    printed_page: str | None
    text: str
    character_count: int


@dataclass(frozen=True)
class SectionRecord:
    section_id: str
    title: str
    start_pdf_page: int
    end_pdf_page: int


@dataclass(frozen=True)
class ChunkRecord:
    chunk_id: str
    section_id: str
    section_title: str
    start_pdf_page: int
    end_pdf_page: int
    text: str
    character_count: int


_SECTION_PATTERNS = (
    ("executive_summary", "Executive Summary", re.compile(r"\bExecutive Summary\b", re.I)),
    ("chapter_1", "Planning the Opportunities for Youth", re.compile(r"\bCHAPTER\s*1\b", re.I)),
    ("chapter_2", "Youth Well-Being and the Demographic Dividend", re.compile(r"\bCHAPTER\s*2\b", re.I)),
    ("chapter_3", "The Economic Inclusion of Youth", re.compile(r"\bCHAPTER\s*3\b", re.I)),
    ("chapter_4", "Education and Training in the New Economy", re.compile(r"\bCHAPTER\s*4\b", re.I)),
    ("chapter_5", "The Fourth Industrial Revolution: Embracing Technology", re.compile(r"\bCHAPTER\s*5\b", re.I)),
    ("chapter_6", "Public Investment in Youth", re.compile(r"\bCHAPTER\s*6\b", re.I)),
    ("appendices", "Appendices", re.compile(r"\bAPPENDIX\s+A\b", re.I)),
)


def _normalise_line(line: str) -> str:
    return re.sub(r"\s+", " ", line).strip()


def _printed_page(text: str) -> str | None:
    first_line = text.splitlines()[0] if text.splitlines() else ""
    match = re.match(r"\s*([ivxlcdm]+|\d+)\s+(?:\||$)", first_line, re.I)
    return match.group(1) if match else None


def _repeated_margin_lines(raw_pages: list[str], minimum_fraction: float = 0.18) -> set[str]:
    """Identify repeated short header/footer lines without hard-coding report text."""

    counts: Counter[str] = Counter()
    for text in raw_pages:
        lines = [_normalise_line(line) for line in text.splitlines() if line.strip()]
        candidates = lines[:2] + lines[-2:]
        counts.update(set(line for line in candidates if 8 <= len(line) <= 140))
    threshold = max(4, int(len(raw_pages) * minimum_fraction))
    return {line for line, count in counts.items() if count >= threshold}


def clean_page_text(text: str, repeated_lines: set[str] | None = None) -> str:
    """Repair common PDF artefacts while preserving paragraph boundaries."""

    repeated_lines = repeated_lines or set()
    text = text.replace("\u00ad", "").replace("\x00", "")
    lines = []
    for raw_line in text.splitlines():
        line = _normalise_line(raw_line)
        if not line or line in repeated_lines:
            continue
        lines.append(line)
    text = "\n".join(lines)
    text = re.sub(r"(?<=\w)-\n(?=[a-z])", "", text)
    text = re.sub(r"(?<![.!?:;])\n(?=[a-z0-9(])", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_pages(pdf_path: Path) -> list[PageRecord]:
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    reader = PdfReader(str(pdf_path))
    raw_pages = [(page.extract_text() or "") for page in reader.pages]
    repeated = _repeated_margin_lines(raw_pages)
    return [
        PageRecord(
            pdf_page=index,
            printed_page=_printed_page(raw),
            text=clean_page_text(raw, repeated),
            character_count=len(clean_page_text(raw, repeated)),
        )
        for index, raw in enumerate(raw_pages, start=1)
    ]


def detect_sections(pages: list[PageRecord]) -> list[SectionRecord]:
    """Find the first occurrence of each major report heading."""

    starts: list[tuple[str, str, int]] = []
    for section_id, title, pattern in _SECTION_PATTERNS:
        candidates = [
            page
            for page in pages
            if "Contents" not in page.text[:250] and pattern.search(page.text)
        ]
        if section_id.startswith("chapter_"):
            # NHDR chapter cover pages contain the heading and little else. Mentions
            # in the chapter-one outline and endnotes are much longer pages.
            cover_pages = [page for page in candidates if page.character_count < 1_000]
            match_page = cover_pages[0].pdf_page if cover_pages else None
        elif section_id == "appendices":
            # Appendix A is mentioned earlier in prose; on its real opening page
            # the heading occurs near the beginning of the extracted text.
            opening_pages = [
                page
                for page in candidates
                if pattern.search(page.text) is not None
                and pattern.search(page.text).start() < 400
            ]
            match_page = opening_pages[0].pdf_page if opening_pages else None
        else:
            match_page = candidates[0].pdf_page if candidates else None
        if match_page is not None:
            starts.append((section_id, title, match_page))
    starts.sort(key=lambda item: item[2])
    sections = []
    for index, (section_id, title, start) in enumerate(starts):
        end = starts[index + 1][2] - 1 if index + 1 < len(starts) else len(pages)
        sections.append(SectionRecord(section_id, title, start, end))
    return sections


def _page_blocks(pages: Iterable[PageRecord]) -> list[tuple[int, str]]:
    blocks: list[tuple[int, str]] = []
    for page in pages:
        # Paragraph-level blocks let overlap retain a useful tail without
        # duplicating an entire 4,000-character PDF page.
        paragraphs = [part.strip() for part in re.split(r"\n+", page.text) if part.strip()]
        blocks.extend((page.pdf_page, paragraph) for paragraph in paragraphs)
    return blocks


def create_chunks(
    pages: list[PageRecord],
    sections: list[SectionRecord],
    chunk_size: int = 5_500,
    overlap: int = 650,
) -> list[ChunkRecord]:
    """Create page-aware overlapping chunks that retain source provenance."""

    if chunk_size <= 0 or not 0 <= overlap < chunk_size:
        raise ValueError("Require chunk_size > 0 and 0 <= overlap < chunk_size")
    chunks: list[ChunkRecord] = []
    page_lookup = {page.pdf_page: page for page in pages}
    for section in sections:
        blocks = _page_blocks(
            page_lookup[number]
            for number in range(section.start_pdf_page, section.end_pdf_page + 1)
            if number in page_lookup
        )
        cursor = 0
        local_index = 1
        while cursor < len(blocks):
            selected: list[tuple[int, str]] = []
            count = 0
            next_cursor = cursor
            while next_cursor < len(blocks):
                page_num, text = blocks[next_cursor]
                addition = len(text) + (2 if selected else 0)
                if selected and count + addition > chunk_size:
                    break
                selected.append((page_num, text))
                count += addition
                next_cursor += 1
                if count >= chunk_size:
                    break
            if not selected:
                page_num, text = blocks[cursor]
                selected = [(page_num, text[:chunk_size])]
                next_cursor = cursor + 1
            chunk_text = "\n\n".join(text for _, text in selected)
            chunks.append(
                ChunkRecord(
                    chunk_id=f"{section.section_id}_{local_index:03d}",
                    section_id=section.section_id,
                    section_title=section.title,
                    start_pdf_page=selected[0][0],
                    end_pdf_page=selected[-1][0],
                    text=chunk_text,
                    character_count=len(chunk_text),
                )
            )
            if next_cursor >= len(blocks):
                break
            retained = 0
            overlap_cursor = next_cursor
            while overlap_cursor > cursor + 1 and retained < overlap:
                overlap_cursor -= 1
                retained += len(blocks[overlap_cursor][1])
            cursor = max(cursor + 1, overlap_cursor)
            local_index += 1
    return chunks


def extract_tables(pdf_path: Path, minimum_rows: int = 1) -> list[dict]:
    """Capture raw tables with PDF page provenance using pdfplumber."""

    import pdfplumber

    records: list[dict] = []
    with pdfplumber.open(str(pdf_path)) as document:
        for pdf_page, page in enumerate(document.pages, start=1):
            for table_index, table in enumerate(page.extract_tables(), start=1):
                rows = [
                    [(_normalise_line(cell) if cell else None) for cell in row]
                    for row in table
                    if any(cell and str(cell).strip() for cell in row)
                ]
                if len(rows) >= minimum_rows:
                    records.append(
                        {
                            "table_id": f"p{pdf_page:03d}_t{table_index:02d}",
                            "pdf_page": pdf_page,
                            "rows": rows,
                        }
                    )
    return records


def records_to_dicts(records: Iterable[object]) -> list[dict]:
    return [asdict(record) for record in records]
