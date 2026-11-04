"""Document loading and text extraction engine.

Supports PDF, DOCX, TXT, and Markdown files with page-level tracking,
metadata extraction, and life-stage folder inference.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar

import docx
import pymupdf

from female_wellness.core.constants import PRIMARY_LIFE_STAGES, LifeStageId
from female_wellness.core.exceptions import DocumentIngestionError
from female_wellness.core.logging_config import get_logger
from female_wellness.schemas.documents import DocumentMetadata, LoadedDocument, RawPage
from female_wellness.utils.text_utils import clean_text, compute_content_hash

logger = get_logger("rag.document_loader")


class BaseLoader(ABC):
    """Abstract base class for format-specific document extractors."""

    @abstractmethod
    def extract_pages(self, file_path: Path) -> list[RawPage]:
        """Extract pages or sections from the target document file.

        Args:
            file_path: Absolute or relative path to the file.

        Returns:
            List of RawPage instances with 1-based page numbers.
        """


class PDFLoader(BaseLoader):
    """Extracts text and page-level metadata from PDF files using PyMuPDF."""

    def extract_pages(self, file_path: Path) -> list[RawPage]:
        pages: list[RawPage] = []
        try:
            with pymupdf.open(file_path) as doc:
                for page_idx, page in enumerate(doc):
                    page_text = page.get_text("text")
                    cleaned = clean_text(page_text)
                    if cleaned:
                        pages.append(RawPage(page_number=page_idx + 1, text=cleaned))
        except Exception as exc:
            raise DocumentIngestionError(
                f"Failed to extract text from PDF '{file_path.name}': {exc}",
                details={"file": str(file_path), "error": str(exc)},
            ) from exc

        return pages


class DocxLoader(BaseLoader):
    """Extracts text from DOCX files using python-docx."""

    def extract_pages(self, file_path: Path) -> list[RawPage]:
        try:
            doc = docx.Document(file_path)
            paragraphs: list[str] = []

            # Extract body paragraphs
            for p in doc.paragraphs:
                cleaned = clean_text(p.text)
                if cleaned:
                    paragraphs.append(cleaned)

            # Extract table cell contents
            for table in doc.tables:
                for row in table.rows:
                    row_text = " | ".join(
                        clean_text(cell.text) for cell in row.cells if cell.text.strip()
                    )
                    if row_text:
                        paragraphs.append(row_text)

            full_text = "\n\n".join(paragraphs)
            if not full_text:
                return []

            # DOCX has no fixed page boundaries; treat as single page (or section)
            return [RawPage(page_number=1, text=full_text)]
        except Exception as exc:
            raise DocumentIngestionError(
                f"Failed to extract text from DOCX '{file_path.name}': {exc}",
                details={"file": str(file_path), "error": str(exc)},
            ) from exc


class TextLoader(BaseLoader):
    """Extracts text from plain text files."""

    def extract_pages(self, file_path: Path) -> list[RawPage]:
        try:
            content = file_path.read_text(encoding="utf-8", errors="replace")
            cleaned = clean_text(content)
            if not cleaned:
                return []
            return [RawPage(page_number=1, text=cleaned)]
        except Exception as exc:
            raise DocumentIngestionError(
                f"Failed to read TXT file '{file_path.name}': {exc}",
                details={"file": str(file_path), "error": str(exc)},
            ) from exc


class MarkdownLoader(BaseLoader):
    """Extracts text from Markdown files."""

    def extract_pages(self, file_path: Path) -> list[RawPage]:
        try:
            content = file_path.read_text(encoding="utf-8", errors="replace")
            cleaned = clean_text(content)
            if not cleaned:
                return []
            return [RawPage(page_number=1, text=cleaned)]
        except Exception as exc:
            raise DocumentIngestionError(
                f"Failed to read MD file '{file_path.name}': {exc}",
                details={"file": str(file_path), "error": str(exc)},
            ) from exc


class DocumentLoader:
    """Unified document loading coordinator with format dispatch and metadata inference."""

    _LOADERS: ClassVar[dict[str, type[BaseLoader]]] = {
        ".pdf": PDFLoader,
        ".docx": DocxLoader,
        ".txt": TextLoader,
        ".md": MarkdownLoader,
    }

    @classmethod
    def infer_life_stage_from_path(cls, file_path: Path) -> str:
        """Infer target life stage from folder hierarchy.

        Example:
            knowledge_base/raw/cycle_fertility/guide.pdf -> 'cycle_fertility'

        Args:
            file_path: Path to the document.

        Returns:
            Recognized LifeStageId value or 'general'.
        """
        for parent in file_path.parents:
            parent_name = parent.name.lower()
            for stage in PRIMARY_LIFE_STAGES:
                if stage.value == parent_name:
                    return stage.value
            if parent_name == LifeStageId.GENERAL.value:
                return LifeStageId.GENERAL.value

        return LifeStageId.GENERAL.value

    @classmethod
    def infer_category_from_text(cls, text: str) -> str:
        """Infer category from header in document text if present."""
        import re
        match = re.search(r"(?:\*\*Category:\*\*|Category:)\s*([a-zA-Z0-9_]+)", text, re.IGNORECASE)
        if match:
            return match.group(1).lower().strip()
        return "general"

    def load_file(
        self,
        file_path: Path | str,
        life_stage: str | None = None,
        category: str | None = None,
    ) -> LoadedDocument:
        """Load a file, extract its text content, and construct a LoadedDocument model.

        Args:
            file_path: Path to the source file.
            life_stage: Explicit life stage override, or None to infer from path.
            category: Domain category override, or None to default to 'general'.

        Returns:
            LoadedDocument with extracted pages and verified metadata.

        Raises:
            DocumentIngestionError: If file does not exist, format is unsupported, or text is empty.
        """
        path = Path(file_path).resolve()
        if not path.is_file():
            raise DocumentIngestionError(f"Source file does not exist: {path}")

        ext = path.suffix.lower()
        loader_cls = self._LOADERS.get(ext)
        if not loader_cls:
            supported = ", ".join(self._LOADERS.keys())
            raise DocumentIngestionError(
                f"Unsupported document format '{ext}' for file '{path.name}'. Supported: {supported}"
            )

        # 1. Read binary content for hash computation
        file_bytes = path.read_bytes()
        file_hash = compute_content_hash(file_bytes)

        # 2. Extract text pages
        loader = loader_cls()
        pages = loader.extract_pages(path)

        if not pages:
            raise DocumentIngestionError(
                f"Document '{path.name}' yielded no extractable text content."
            )

        # 3. Assemble aggregated full text
        full_text = "\n\n".join(p.text for p in pages)

        # 4. Infer or assign metadata
        assigned_stage = life_stage or self.infer_life_stage_from_path(path)
        assigned_category = category or self.infer_category_from_text(full_text)
        doc_id = f"doc_{file_hash[:12]}"

        metadata = DocumentMetadata(
            document_id=doc_id,
            source=path.name,
            file_type=ext.lstrip("."),  # type: ignore[arg-type]
            life_stage=assigned_stage,
            category=assigned_category,
            title=path.stem.replace("_", " ").title(),
            total_pages=len(pages),
            file_hash=file_hash,
        )

        logger.debug(
            "Successfully loaded document '%s' (%d pages, stage='%s')",
            path.name,
            len(pages),
            assigned_stage,
        )

        return LoadedDocument(metadata=metadata, pages=pages, full_text=full_text)

