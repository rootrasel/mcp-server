"""Unit tests for document loading, format extraction, and metadata inference."""

from pathlib import Path

import docx
import pymupdf
import pytest

from female_wellness.core.exceptions import DocumentIngestionError
from female_wellness.rag.document_loader import DocumentLoader


@pytest.fixture
def sample_files(tmp_path: Path) -> dict[str, Path]:
    """Create temporary test files in different formats."""
    # 1. Plain text file in beauty_radiance folder
    beauty_dir = tmp_path / "knowledge_base" / "raw" / "beauty_radiance"
    beauty_dir.mkdir(parents=True)
    txt_file = beauty_dir / "skin_hydration.txt"
    txt_file.write_text(
        "Adequate cellular hydration supports the epidermal barrier. "
        "Hyaluronic acid and ceramides work synergistically to maintain moisture.",
        encoding="utf-8",
    )

    # 2. Markdown file in cycle_fertility folder
    cycle_dir = tmp_path / "knowledge_base" / "raw" / "cycle_fertility"
    cycle_dir.mkdir(parents=True)
    md_file = cycle_dir / "luteal_phase.md"
    md_file.write_text(
        "# Luteal Phase Nutrition\n\n"
        "During the luteal phase, basal body temperature rises slightly due to progesterone. "
        "Magnesium and complex carbohydrates support steady energy.",
        encoding="utf-8",
    )

    # 3. DOCX file in athlete_performance folder
    athlete_dir = tmp_path / "knowledge_base" / "raw" / "athlete_performance"
    athlete_dir.mkdir(parents=True)
    docx_file = athlete_dir / "hrv_training.docx"
    doc = docx.Document()
    doc.add_heading("HRV & Female Training Periodization", level=1)
    doc.add_paragraph("High HRV generally reflects robust parasympathetic recovery.")
    doc.save(docx_file)

    # 4. PDF file in perimenopause folder
    peri_dir = tmp_path / "knowledge_base" / "raw" / "perimenopause"
    peri_dir.mkdir(parents=True)
    pdf_file = peri_dir / "vasomotor_guide.pdf"
    pdf_doc = pymupdf.open()
    page1 = pdf_doc.new_page()
    page1.insert_text(
        (50, 50),
        "Page 1: Vasomotor symptoms like hot flashes occur due to hypothalamic estrogen fluctuations.",
    )
    page2 = pdf_doc.new_page()
    page2.insert_text(
        (50, 50),
        "Page 2: Phytoestrogens and resistance training support bone and cardiovascular vitality.",
    )
    pdf_doc.save(pdf_file)
    pdf_doc.close()

    return {
        "txt": txt_file,
        "md": md_file,
        "docx": docx_file,
        "pdf": pdf_file,
    }


def test_load_text_file(sample_files):
    """Verify loading and life-stage inference for text files."""
    loader = DocumentLoader()
    doc = loader.load_file(sample_files["txt"])

    assert doc.metadata.source == "skin_hydration.txt"
    assert doc.metadata.file_type == "txt"
    assert doc.metadata.life_stage == "beauty_radiance"
    assert "cellular hydration" in doc.full_text
    assert len(doc.pages) == 1
    assert doc.pages[0].page_number == 1


def test_load_markdown_file(sample_files):
    """Verify loading and text extraction for markdown files."""
    loader = DocumentLoader()
    doc = loader.load_file(sample_files["md"])

    assert doc.metadata.source == "luteal_phase.md"
    assert doc.metadata.life_stage == "cycle_fertility"
    assert "progesterone" in doc.full_text


def test_load_docx_file(sample_files):
    """Verify loading and paragraph extraction for DOCX files."""
    loader = DocumentLoader()
    doc = loader.load_file(sample_files["docx"])

    assert doc.metadata.source == "hrv_training.docx"
    assert doc.metadata.life_stage == "athlete_performance"
    assert "parasympathetic recovery" in doc.full_text


def test_load_pdf_file_with_pages(sample_files):
    """Verify page-by-page extraction for PDF files."""
    loader = DocumentLoader()
    doc = loader.load_file(sample_files["pdf"])

    assert doc.metadata.source == "vasomotor_guide.pdf"
    assert doc.metadata.life_stage == "perimenopause"
    assert len(doc.pages) == 2
    assert doc.pages[0].page_number == 1
    assert "Page 1: Vasomotor symptoms" in doc.pages[0].text
    assert doc.pages[1].page_number == 2
    assert "Page 2: Phytoestrogens" in doc.pages[1].text


def test_unsupported_format(tmp_path):
    """Verify error on unsupported format."""
    dummy_file = tmp_path / "invalid.xyz"
    dummy_file.write_text("Hello")
    loader = DocumentLoader()

    with pytest.raises(DocumentIngestionError) as exc_info:
        loader.load_file(dummy_file)
    assert "Unsupported document format" in str(exc_info.value)


def test_nonexistent_file():
    """Verify error when file does not exist."""
    loader = DocumentLoader()
    with pytest.raises(DocumentIngestionError) as exc_info:
        loader.load_file("nonexistent_path_12345.pdf")
    assert "does not exist" in str(exc_info.value)

