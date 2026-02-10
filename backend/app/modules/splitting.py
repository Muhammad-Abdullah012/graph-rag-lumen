"""Text Splitting Module"""
import logging
from typing import List, Tuple

from langchain.text_splitter import MarkdownHeaderTextSplitter

logger = logging.getLogger(__name__)


class TextSplitter:
    """Split extracted markdown text by headers and structure"""
    
    def __init__(self):
        """Initialize text splitter with header hierarchy"""
        self.headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
            ("####", "Header 4"),
        ]
        
        self.splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=self.headers_to_split_on,
            return_each_line=False,
            strip_headers=False
        )
    
    def split(
        self,
        text: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ) -> List[Tuple[str, dict]]:
        """
        Split markdown text by headers while preserving structure
        
        Args:
            text: Markdown text to split
            chunk_size: Target size of each chunk
            chunk_overlap: Overlap between chunks
            
        Returns:
            List of tuples (chunk_text, metadata_dict)
        """
        try:
            logger.info(f"Splitting text with chunk_size={chunk_size}")
            
            # First split by headers to preserve structure
            splits = self.splitter.split_text(text)
            
            chunks_with_metadata = []
            
            for split in splits:
                chunk_text = split.page_content
                metadata = split.metadata
                
                # If chunk is too large, split further
                if len(chunk_text) > chunk_size:
                    sub_chunks = self._split_large_chunk(
                        chunk_text,
                        chunk_size,
                        chunk_overlap
                    )
                    for sub_chunk in sub_chunks:
                        chunks_with_metadata.append((sub_chunk, metadata))
                else:
                    chunks_with_metadata.append((chunk_text, metadata))
            
            logger.info(f"Created {len(chunks_with_metadata)} text chunks")
            return chunks_with_metadata
            
        except Exception as e:
            logger.error(f"Error splitting text: {str(e)}")
            raise
    
    def _split_large_chunk(
        self,
        text: str,
        chunk_size: int,
        overlap: int
    ) -> List[str]:
        """Split a large chunk into smaller overlapping pieces"""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])
            start = end - overlap
        
        return chunks
