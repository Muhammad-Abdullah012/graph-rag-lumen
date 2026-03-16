"""OCR Pipeline using Mistral OCR API"""
import base64
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from mistralai.client import Mistral

from config.settings import settings

logger = logging.getLogger(__name__)


class MistralOCRPipeline:
    """Extract structured page data from PDFs using Mistral OCR"""

    def __init__(self):
        self.client = Mistral(api_key=settings.mistral_api_key)
        Path(settings.json_output_path).mkdir(parents=True, exist_ok=True)
        Path(settings.images_path).mkdir(parents=True, exist_ok=True)

    def process(self, pdf_path: str, document_id: str, filename: str) -> list[dict]:
        """
        Run Mistral OCR on a PDF and return structured page data.

        Args:
            pdf_path: Absolute path to the PDF file
            document_id: Unique document identifier (UUID)
            filename: Original filename including UUID prefix (e.g. "{uuid}_{name}.pdf")

        Returns:
            List of page dicts with index, markdown, images, tables, hyperlinks,
            header, footer, dimensions
        """
        source_file = self._strip_uuid_prefix(filename)
        logger.info(f"Starting OCR for {source_file} (id={document_id})")

        doc_url = self._encode_pdf(pdf_path)
        pages = self._run_ocr(doc_url)
        pages = self._save_images(pages, document_id)
        self._save_json(pages, document_id, source_file)

        logger.info(f"OCR complete for {source_file}: {len(pages)} pages")
        return pages

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _strip_uuid_prefix(self, filename: str) -> str:
        """Remove the leading '{uuid}_' from a stored filename."""
        parts = filename.split("_", 1)
        return parts[1] if len(parts) == 2 else filename

    def _encode_pdf(self, pdf_path: str) -> str:
        """Return data-URI for the PDF."""
        with open(pdf_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        return f"data:application/pdf;base64,{b64}"

    def _run_ocr(self, doc_url: str) -> list[dict]:
        """Call the synchronous OCR endpoint and return list of page dicts."""
        response = self.client.ocr.process(
            model="mistral-ocr-latest",
            document={
                "type": "document_url",
                "document_url": doc_url,
            },
            include_image_base64=True,
            table_format="markdown",
            extract_header=True,
            extract_footer=True,
        )
        return [p.model_dump() for p in response.pages]

    def _save_images(self, pages: list[dict], document_id: str) -> list[dict]:
        """
        Save base64 images to disk, replace image_base64 with file_path in each
        image dict, and return the updated pages list.
        """
        for page in pages:
            page_index = page.get("index", 0)
            images = page.get("images") or []
            updated_images = []
            for img in images:
                img = dict(img)  # shallow copy
                b64_data = img.pop("image_base64", None)
                if b64_data:
                    img_filename = f"{document_id}_p{page_index}_{img['id']}"
                    img_path = Path(settings.images_path) / img_filename
                    img_path.write_bytes(base64.b64decode(b64_data))
                    img["file_path"] = str(img_path)
                    logger.debug(f"Saved image: {img_path}")
                updated_images.append(img)
            page["images"] = updated_images
        return pages

    def _save_json(self, pages: list[dict], document_id: str, source_file: str):
        """Persist the full page data to a JSON file."""
        output = {
            "document_id": document_id,
            "source_file": source_file,
            "processed_at": datetime.now(timezone.utc).isoformat(),
            "pages": pages,
        }
        json_path = Path(settings.json_output_path) / f"{Path(source_file).stem}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved OCR JSON: {json_path}")


# Singleton
_ocr_pipeline: MistralOCRPipeline | None = None


def get_ocr_pipeline() -> MistralOCRPipeline:
    global _ocr_pipeline
    if _ocr_pipeline is None:
        _ocr_pipeline = MistralOCRPipeline()
    return _ocr_pipeline
