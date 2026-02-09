"""PDF Extraction Module using Docling"""
import logging
import multiprocessing
from pathlib import Path
from typing import Optional

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    AcceleratorOptions,
    PdfPipelineOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption

logger = logging.getLogger(__name__)


class PDFExtractor:
    """Extract text from PDF documents using Docling"""
    
    def __init__(self):
        """Initialize PDF extractor with optimal settings"""
        self.pdf_options = PdfPipelineOptions(
            do_code_enrichment=True,
            do_ocr=True,
            do_table_structure=True,
            do_picture_classification=True,
            accelerator_options=AcceleratorOptions(
                num_threads=multiprocessing.cpu_count(),
                device='auto'
            )
        )
        
        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=self.pdf_options
                )
            }
        )
    
    def extract(self, pdf_path: str) -> str:
        """
        Extract markdown text from PDF
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Extracted text as markdown
            
        Raises:
            FileNotFoundError: If PDF file doesn't exist
            Exception: If extraction fails
        """
        try:
            pdf_file = Path(pdf_path)
            
            if not pdf_file.exists():
                raise FileNotFoundError(f"PDF file not found: {pdf_path}")
            
            if not pdf_file.suffix.lower() == '.pdf':
                raise ValueError(f"File is not a PDF: {pdf_path}")
            
            logger.info(f"Extracting text from: {pdf_path}")
            
            result = self.converter.convert(str(pdf_path))
            processed_text = result.document.export_to_markdown()
            
            logger.info(f"Successfully extracted {len(processed_text)} characters from {pdf_file.name}")
            
            return processed_text
            
        except Exception as e:
            logger.error(f"Error extracting PDF {pdf_path}: {str(e)}")
            raise


# Singleton instance
_extractor = None


def get_pdf_extractor() -> PDFExtractor:
    """Get or create PDF extractor instance"""
    global _extractor
    if _extractor is None:
        _extractor = PDFExtractor()
    return _extractor
