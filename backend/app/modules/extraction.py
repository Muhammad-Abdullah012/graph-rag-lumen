"""PDF Extraction Module using Docling"""
import logging
import multiprocessing
from pathlib import Path
from typing import Optional

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import ThreadedPdfPipelineOptions, RapidOcrOptions
from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.pipeline.threaded_standard_pdf_pipeline import ThreadedStandardPdfPipeline
from docling.document_converter import DocumentConverter, PdfFormatOption

logger = logging.getLogger(__name__)


class PDFExtractor:
    """Extract text from PDF documents using Docling"""
    
    def __init__(self):
        """Initialize PDF extractor with optimal settings"""
        accelerator_options = AcceleratorOptions(
            device=AcceleratorDevice.CUDA,  # Use CUDA for NVIDIA GPUs
        )
        pdf_options = ThreadedPdfPipelineOptions(
            do_code_enrichment=True,
            do_ocr=True,
            ocr_options=RapidOcrOptions(
                backend="torch",
            ),
            do_table_structure=True,
            do_picture_classification=True,
            do_formula_enrichment=True,
            ocr_batch_size=8,
            layout_batch_size=8,
            table_batch_size=2,
            accelerator_options=accelerator_options,
        )
        
        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_cls=ThreadedStandardPdfPipeline,
                    pipeline_options=pdf_options
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
