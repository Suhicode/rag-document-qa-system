"""
Document loader with multi-format support (PDF, TXT, DOCX, PPTX, XLSX, CSV, MD, HTML).
"""

import os
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
import csv
import io

from langchain_community.document_loaders import (
    PyPDFLoader, TextLoader,
    Docx2txtLoader, UnstructuredPowerPointLoader,
    UnstructuredMarkdownLoader, CSVLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

from chunking_adviser import ChunkingAdviser, ChunkingDecision


def normalize_text(text: str) -> str:
    """
    Normalize extracted text to fix PDF extraction artifacts.
    
    Fixes:
    - Character-spaced words: "e n v i r o n m e n t s" → "environments"
    - Excessive whitespace: "Y O U S U F" → "YOUSUF" (then handled normally)
    - Mixed spacing issues from PDF extraction
    
    Args:
        text: Raw extracted text
        
    Returns:
        Normalized text
    """
    if not text:
        return text
    
    # Pattern 1: Single-character words separated by single spaces
    # e.g., "e n v i r o n m e n t s" → "environments"
    # Match sequences of single letters separated by single spaces
    def rejoin_spaced_chars(match):
        chars = match.group(0)
        # Remove all spaces between characters
        return chars.replace(' ', '')
    
    # Find sequences of 3+ single characters separated by single spaces
    # This handles "e n v" but not "word  word" (double space)
    text = re.sub(r'(?:\b\w\b(?: \b\w\b){2,})', rejoin_spaced_chars, text)
    
    # Pattern 2: Collapse multiple whitespace to single space
    text = re.sub(r'\s+', ' ', text)
    
    # Pattern 3: Fix spacing around punctuation
    text = re.sub(r'\s+([.,!?;:])', r'\1', text)
    text = re.sub(r'([.,!?;:])\s*', r'\1 ', text)
    
    # Pattern 4: Trim leading/trailing whitespace
    text = text.strip()
    
    return text


class DocumentLoader:
    """Loads and chunks documents in multiple formats."""
    
    # Supported file extensions with their loaders
    SUPPORTED_FORMATS = {
        '.pdf': 'pdf',
        '.txt': 'text',
        '.docx': 'docx',
        '.doc': 'docx',
        '.pptx': 'pptx',
        '.ppt': 'pptx',
        '.csv': 'csv',
        '.xlsx': 'excel',
        '.xls': 'excel',
        '.md': 'markdown',
        '.markdown': 'markdown',
        '.html': 'html',
        '.htm': 'html'
    }
    
    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        chunking_strategy: str = "recursive",
        auto_chunking: bool = False,
        filename: str = ""
    ):
        """
        Initialize the document loader.
        
        Args:
            chunk_size: Target size of each chunk
            chunk_overlap: Overlap between chunks
            chunking_strategy: "recursive" or "token" based
            auto_chunking: If True, use Chunking Adviser to auto-select parameters
            filename: Original filename (needed for auto-chunking detection)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunking_strategy = chunking_strategy
        self.auto_chunking = auto_chunking
        self.filename = filename
        self._adviser_decision: Optional[ChunkingDecision] = None
        
        if chunking_strategy == "recursive":
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
        else:
            from langchain.text_splitter import CharacterTextSplitter
            self.text_splitter = CharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separator="\n"
            )
    
    def load_document(self, file_path: str) -> List[Document]:
        """
        Load a document from file path with format auto-detection.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            List of LangChain Document objects
        """
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Document not found: {file_path}")
        
        extension = path.suffix.lower()
        
        if extension not in self.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported file type: {extension}. "
                f"Supported: {', '.join(self.SUPPORTED_FORMATS.keys())}"
            )
        
        # Get appropriate loader
        loader_type = self.SUPPORTED_FORMATS[extension]
        loader = self._get_loader(loader_type, str(path))
        
        documents = loader.load()
        
        # Normalize text to fix PDF extraction artifacts
        for doc in documents:
            doc.page_content = normalize_text(doc.page_content)
        
        # Add source filename and format to metadata
        for doc in documents:
            doc.metadata["source_file"] = path.name
            doc.metadata["file_type"] = extension
        
        return documents
    
    def _get_loader(self, loader_type: str, file_path: str):
        """
        Get the appropriate loader based on file type.
        
        BONUS: Multiple format support
        """
        loaders = {
            'pdf': PyPDFLoader,
            'text': lambda p: TextLoader(p, encoding="utf-8"),
            'docx': Docx2txtLoader,
            'pptx': UnstructuredPowerPointLoader,
            'markdown': UnstructuredMarkdownLoader,
            'csv': lambda p: CSVLoader(p, encoding="utf-8"),
            'excel': self._load_excel,
            'html': self._load_html
        }
        
        if loader_type in loaders:
            loader_class = loaders[loader_type]
            if callable(loader_class) and not isinstance(loader_class, type):
                return loader_class(file_path)
            return loader_class(file_path)
        else:
            raise ValueError(f"No loader available for type: {loader_type}")
    
    def _load_excel(self, file_path: str) -> List[Document]:
        """
        Load Excel files using pandas.
        
        BONUS: Excel file support
        """
        try:
            import pandas as pd
            df = pd.read_excel(file_path)
            
            # Convert each row to a document
            documents = []
            for idx, row in df.iterrows():
                content = " | ".join([f"{col}: {val}" for col, val in row.items() if pd.notna(val)])
                if content.strip():
                    documents.append(Document(
                        page_content=content,
                        metadata={"row": idx, "source_file": Path(file_path).name}
                    ))
            
            return documents
        except ImportError:
            raise ImportError("pandas required for Excel support. Install: pip install pandas openpyxl")
    
    def _load_html(self, file_path: str) -> List[Document]:
        """
        Load HTML files using BeautifulSoup.
        
        BONUS: HTML file support
        """
        try:
            from bs4 import BeautifulSoup
            
            with open(file_path, 'r', encoding='utf-8') as f:
                soup = BeautifulSoup(f.read(), 'html.parser')
            
            # Extract text, removing script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            text = soup.get_text(separator='\n', strip=True)
            
            return [Document(
                page_content=text,
                metadata={"source_file": Path(file_path).name}
            )]
        except ImportError:
            raise ImportError("beautifulsoup4 required for HTML support. Install: pip install beautifulsoup4")
    
    def apply_chunking_advice(self, raw_text: str) -> None:
        """
        Apply Chunking Adviser recommendations to adjust parameters.
        
        Called automatically before chunking if auto_chunking is enabled.
        """
        if not self.auto_chunking:
            return
        
        adviser = ChunkingAdviser()
        decision = adviser.decide_chunking(
            raw_text=raw_text,
            filename=self.filename,
            manual_size=self.chunk_size if not self.auto_chunking else None,
            manual_overlap=self.chunk_overlap if not self.auto_chunking else None
        )
        
        self._adviser_decision = decision
        self.chunk_size = decision.chunk_size
        self.chunk_overlap = decision.chunk_overlap
        
        # Recreate text splitter with new parameters
        if self.chunking_strategy == "recursive":
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
    
    def get_chunking_decision(self) -> Optional[ChunkingDecision]:
        """Return the chunking adviser decision if available."""
        return self._adviser_decision
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into semantic chunks."""
        if not documents:
            return []
        
        # Apply auto-chunking if enabled
        if self.auto_chunking:
            raw_text = "\n\n".join([doc.page_content for doc in documents])
            self.apply_chunking_advice(raw_text)
        
        return self.text_splitter.split_documents(documents)
    
    def load_and_chunk(self, file_path: str) -> List[Document]:
        """Convenience method: load and chunk in one step."""
        documents = self.load_document(file_path)
        chunks = self.chunk_documents(documents)
        
        # Show chunking strategy if auto-chunking was used
        if self.auto_chunking and self._adviser_decision:
            decision = self._adviser_decision
            print(f"📦 Chunking Strategy: Size={decision.chunk_size} | Overlap={decision.chunk_overlap} | {decision.document_type.upper()}")
        
        print(f"Loaded {file_path}: {len(documents)} pages → {len(chunks)} chunks")
        return chunks


def validate_document_path(file_path: str) -> bool:
    """Validate that a file path is valid and supported."""
    path = Path(file_path)
    
    if not path.exists():
        print(f"Error: File not found: {file_path}")
        return False
    
    extension = path.suffix.lower()
    supported = DocumentLoader.SUPPORTED_FORMATS.keys()
    
    if extension not in supported:
        print(f"Error: Unsupported file type: {extension}")
        print(f"Supported types: {', '.join(supported)}")
        return False
    
    return True


def compare_chunking_strategies(
    file_path: str,
    strategies: List[str] = ["recursive", "token"],
    sizes: List[int] = [300, 500, 800],
    overlaps: List[int] = [50, 100, 150]
) -> Dict[str, Any]:
    """
    BONUS: Compare different chunking strategies.
    
    Returns statistics for each configuration.
    """
    results = {}
    
    for strategy in strategies:
        for size in sizes:
            for overlap in overlaps:
                if overlap >= size:
                    continue
                
                try:
                    loader = DocumentLoader(
                        chunk_size=size,
                        chunk_overlap=overlap,
                        chunking_strategy=strategy
                    )
                    chunks = loader.load_and_chunk(file_path)
                    
                    key = f"{strategy}_{size}_{overlap}"
                    results[key] = {
                        "strategy": strategy,
                        "chunk_size": size,
                        "chunk_overlap": overlap,
                        "num_chunks": len(chunks),
                        "avg_chunk_length": sum(len(c.page_content) for c in chunks) / len(chunks) if chunks else 0,
                        "sample_chunk": chunks[0].page_content[:100] if chunks else ""
                    }
                except Exception as e:
                    results[f"{strategy}_{size}_{overlap}"] = {"error": str(e)}
    
    return results
