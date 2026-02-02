"""
Chunking adviser: auto-selects chunk size/overlap based on document type.

Rule-based only. Resumes need small chunks (200), narratives need large (800).
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
import re


class DocumentType(Enum):
    """Document classification for chunking strategy selection."""
    RESUME = "resume"
    TECHNICAL = "technical"
    NARRATIVE = "narrative"
    UNKNOWN = "unknown"


@dataclass
class ChunkingDecision:
    """
    Output of Chunking Adviser containing recommended parameters.
    
    These recommendations optimize chunk boundaries for specific
    document types without changing retrieval safety.
    """
    chunk_size: int
    chunk_overlap: int
    reason: str
    document_type: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "reason": self.reason,
            "document_type": self.document_type
        }


class ChunkingAdviser:
    """
    Recommends chunk size/overlap based on document type.
    
    Resumes: 200/30 (preserves sections)
    Technical: 500/100 (code + explanations)
    Narrative: 800/150 (maintains flow)
    """
    
    # Chunk size recommendations by document type
    CHUNK_SIZES = {
        DocumentType.RESUME: (200, 30),      # (size, overlap)
        DocumentType.TECHNICAL: (500, 100),
        DocumentType.NARRATIVE: (800, 150),
        DocumentType.UNKNOWN: (500, 100),    # Default fallback
    }
    
    # Resume indicators
    RESUME_KEYWORDS = [
        "skills", "experience", "education", "projects", "work",
        "qualifications", "certifications", "achievements",
        "professional", "career", "employment", "internship",
        "contact", "email", "phone", "linkedin", "github"
    ]
    
    # Technical document indicators
    TECHNICAL_KEYWORDS = [
        "api", "function", "method", "class", "module",
        "implementation", "algorithm", "data structure",
        "code", "syntax", "parameter", "return", "import",
        "def", "class", "function", "async", "await",
        "installation", "configuration", "deployment",
        "docker", "kubernetes", "aws", "azure", "gcp"
    ]
    
    def __init__(self):
        """Compile regex patterns."""
        self.bullet_pattern = re.compile(r'^[\s]*[-•*\d+\.]', re.MULTILINE)
        self.heading_pattern = re.compile(r'^[\s]*[A-Z][A-Z\s]{2,}', re.MULTILINE)
    
    def detect_document_type(self, raw_text: str, filename: str = "") -> DocumentType:
        """Detect type from filename, keywords, and structure (bullets/headings)."""
        text_lower = raw_text.lower()
        filename_lower = filename.lower()
        
        # Filename hints (strong signals)
        if any(term in filename_lower for term in ["resume", "cv", "curriculum"]):
            return DocumentType.RESUME
        
        # Keyword scoring
        resume_score = sum(1 for kw in self.RESUME_KEYWORDS if kw in text_lower)
        technical_score = sum(1 for kw in self.TECHNICAL_KEYWORDS if kw in text_lower)
        
        # Structural analysis
        bullet_count = len(self.bullet_pattern.findall(raw_text[:5000]))
        heading_count = len(self.heading_pattern.findall(raw_text[:5000]))
        
        # Resume: high bullet density, section headings
        if resume_score >= 3 or (bullet_count > 5 and heading_count >= 2):
            return DocumentType.RESUME
        
        # Technical: code keywords present
        if technical_score >= 5:
            return DocumentType.TECHNICAL
        
        # Default to narrative if text is long and flowing
        avg_paragraph_length = self._avg_paragraph_length(raw_text)
        if avg_paragraph_length > 200:
            return DocumentType.NARRATIVE
        
        return DocumentType.UNKNOWN
    
    def _avg_paragraph_length(self, text: str) -> float:
        """Calculate average paragraph length to detect narrative style."""
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        if not paragraphs:
            return 0.0
        return sum(len(p) for p in paragraphs[:10]) / len(paragraphs[:10])
    
    def adjust_for_document_length(
        self, 
        base_size: int, 
        base_overlap: int, 
        text_length: int
    ) -> tuple[int, int]:
        """Adjust chunk size for very short (<1k) or very long (>50k) documents."""
        # Very short document (< 1000 chars) - don't over-split
        if text_length < 1000:
            # Use document length as chunk size (essentially no splitting)
            adjusted_size = min(base_size, text_length)
            adjusted_overlap = min(base_overlap, adjusted_size // 4)
            return adjusted_size, adjusted_overlap
        
        # Short document (1-3 pages / ~3000 chars)
        if text_length < 3000:
            # Reduce chunk size proportionally
            ratio = text_length / 3000
            adjusted_size = int(base_size * (0.7 + 0.3 * ratio))
            adjusted_overlap = int(base_overlap * (0.7 + 0.3 * ratio))
            return adjusted_size, adjusted_overlap
        
        # Long document (> 20 pages / ~50000 chars) - larger chunks for efficiency
        if text_length > 50000:
            adjusted_size = min(int(base_size * 1.2), 1000)
            adjusted_overlap = min(int(base_overlap * 1.2), 200)
            return adjusted_size, adjusted_overlap
        
        return base_size, base_overlap
    
    def decide_chunking(
        self,
        raw_text: str,
        filename: str = "",
        manual_size: Optional[int] = None,
        manual_overlap: Optional[int] = None
    ) -> ChunkingDecision:
        """Recommend chunking strategy. Manual params override auto-detection."""
        # Manual override takes precedence
        if manual_size is not None and manual_overlap is not None:
            return ChunkingDecision(
                chunk_size=manual_size,
                chunk_overlap=manual_overlap,
                reason="Manual parameters specified",
                document_type="manual"
            )
        
        # Auto-detect document type
        doc_type = self.detect_document_type(raw_text, filename)
        
        # Get base parameters for this type
        base_size, base_overlap = self.CHUNK_SIZES[doc_type]
        
        # Adjust for document length
        text_length = len(raw_text)
        chunk_size, chunk_overlap = self.adjust_for_document_length(
            base_size, base_overlap, text_length
        )
        
        # Build reasoning
        text_lower = raw_text.lower()
        resume_score = sum(1 for kw in self.RESUME_KEYWORDS if kw in text_lower)
        technical_score = sum(1 for kw in self.TECHNICAL_KEYWORDS if kw in text_lower)
        bullet_count = len(self.bullet_pattern.findall(raw_text[:5000]))
        avg_paragraph_length = self._avg_paragraph_length(raw_text)
        
        if doc_type == DocumentType.RESUME:
            reason = f"Resume-style detected ({resume_score} keywords, {bullet_count} bullets) → small chunks for discrete sections"
        elif doc_type == DocumentType.TECHNICAL:
            reason = f"Technical document detected ({technical_score} keywords) → medium chunks for code + explanations"
        elif doc_type == DocumentType.NARRATIVE:
            reason = f"Narrative style detected ({avg_paragraph_length:.0f} char paragraphs) → large chunks for flow"
        else:
            reason = "Unknown document type → using default parameters"
        
        # Add length adjustment note if applicable
        if text_length < 1000:
            reason += "; shortened for small document"
        elif text_length > 50000:
            reason += "; increased for long document"
        
        return ChunkingDecision(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            reason=reason,
            document_type=doc_type.value
        )
    
    def explain_strategy(self, decision: ChunkingDecision) -> str:
        """Generate human-readable explanation for CLI output."""
        return f"""
📦 Chunking Strategy Selected:
   Size: {decision.chunk_size} characters
   Overlap: {decision.chunk_overlap} characters
   Type: {decision.document_type.upper()}
   
   Why: {decision.reason}
"""


def decide_chunking(
    raw_text: str,
    filename: str = "",
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None
) -> ChunkingDecision:
    """Convenience function for direct usage."""
    adviser = ChunkingAdviser()
    return adviser.decide_chunking(raw_text, filename, chunk_size, chunk_overlap)
