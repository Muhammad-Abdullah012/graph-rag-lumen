import base64
import os
import json
import tempfile
from pydantic import BaseModel, Field
from mistralai import Mistral
from mistralai.models import OCRResponse
from mistralai.extra import response_format_from_pydantic_model
from IPython.display import Markdown, display
from enum import Enum

# ==================== NEW: PDF SPLITTER ====================
from pypdf import PdfReader, PdfWriter

def split_pdf_into_chunks(pdf_path: str, chunk_size: int = 100) -> list[str]:
    """Split PDF into temporary files of max chunk_size pages each."""
    reader = PdfReader(pdf_path)
    num_pages = len(reader.pages)
    chunk_paths = []

    print(f"📄 Total pages in PDF: {num_pages} → Splitting into chunks of {chunk_size} pages...")

    for start in range(0, num_pages, chunk_size):
        end = min(start + chunk_size, num_pages)
        writer = PdfWriter()
        for page_num in range(start, end):
            writer.add_page(reader.pages[page_num])

        tmp_file = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
        writer.write(tmp_file)
        tmp_file.close()
        chunk_paths.append(tmp_file.name)

    print(f"✅ Created {len(chunk_paths)} chunk(s)")
    return chunk_paths


# ==================== YOUR ORIGINAL HELPERS (unchanged) ====================
class ImageType(str, Enum):
    GRAPH = "graph"
    TEXT = "text"
    TABLE = "table"
    IMAGE = "image"

class Image(BaseModel):
    image_type: ImageType = Field(..., description="The type of the image. Must be one of 'graph', 'text', 'table' or 'image'.")
    description: str = Field(..., description="A description of the image.")

class Document(BaseModel):
    language: str = Field(..., description="The language of the document in ISO 639-1 code format (e.g., 'en', 'fr').")
    summary: str = Field(..., description="A summary of the document.")
    authors: list[str] = Field(..., description="A list of authors who contributed to the document.")

def encode_pdf(pdf_path: str) -> str:
    with open(pdf_path, "rb") as pdf_file:
        return base64.b64encode(pdf_file.read()).decode('utf-8')

def replace_images_in_markdown(markdown_str: str, images_dict: dict) -> str:
    for img_name, (base64_str, description) in images_dict.items():
        markdown_str = markdown_str.replace(
            f"![{img_name}]({img_name})",
            f"![{img_name}]({base64_str})\n\n**Description:** {description}"
        )
    return markdown_str

def get_combined_markdown(ocr_response: OCRResponse) -> str:
    markdowns: list[str] = []
    for page in ocr_response.pages:
        image_data = {}
        for img in page.images:
            image_data[img.id] = (img.image_base64, getattr(img, 'description', 'No description available'))
        markdowns.append(replace_images_in_markdown(page.markdown, image_data))
    return "\n\n".join(markdowns)

def save_response_and_markdown(ocr_response: OCRResponse, json_path: str, md_path: str):
    with open(json_path, "w", encoding="utf-8") as json_file:
        json_file.write(ocr_response.model_dump_json(indent=2))

    with open(md_path, "w", encoding="utf-8") as md_file:
        md_file.write(get_combined_markdown(ocr_response))


# ==================== MAIN LOGIC WITH CHUNKING ====================
api_key = os.getenv("MISTRAL_API_KEY")
client = Mistral(api_key=api_key)

pdf_path = r"/content/Handbuch EC 7 Band 1 und 2.PDF"

document_annotation_prompt = """Extract the following from the provided PDF document:

Language (e.g., "English")
All chapter/section titles (e.g., ["Abstract", "1 Introduction"])
All URLs (e.g., ["https://example.com"])

Be precise and include only exact matches."""

# 1. Split into 100-page chunks
chunk_paths = split_pdf_into_chunks(pdf_path, chunk_size=100)

# 2. Process every chunk
all_pages = []
responses = []
markdown_parts = []

for i, chunk_path in enumerate(chunk_paths):
    print(f"🔄 Processing chunk {i+1}/{len(chunk_paths)} ...")
    base64_pdf = encode_pdf(chunk_path)

    ocr_response = client.ocr.process(
        model="mistral-ocr-latest",
        document={
            "type": "document_url",
            "document_url": f"data:application/pdf;base64,{base64_pdf}"
        },
        bbox_annotation_format=response_format_from_pydantic_model(Image),
        document_annotation_format=response_format_from_pydantic_model(Document),
        document_annotation_prompt=document_annotation_prompt,
        include_image_base64=True,
        extract_header=True,
        extract_footer=True,
        table_format="html",
    )

    responses.append(ocr_response)
    all_pages.extend(ocr_response.pages)
    markdown_parts.append(get_combined_markdown(ocr_response))

# 3. Create one combined response (so the JSON looks like a normal full document)
if responses:
    combined_response = responses[0].model_copy(update={"pages": all_pages})
else:
    combined_response = None

# 4. Save combined JSON + Markdown
save_response_and_markdown(combined_response, "ocr_response_combined.json", "output_combined.md")

# 5. Display the full document
display(Markdown("\n\n".join(markdown_parts)))

# 6. Cleanup temporary chunk files
for path in chunk_paths:
    if os.path.exists(path):
        os.unlink(path)

print("🎉 Done! Combined JSON saved as → ocr_response_combined.json")
print("📘 Combined Markdown saved as → output_combined.md")