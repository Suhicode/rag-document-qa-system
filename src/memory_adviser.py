"""
Memory adviser: analyzes queries to optimize retrieval strategy.

Rule-based only - no LLM calls. Doesn't inject content, just adjusts strategy.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum
import re


class DocumentType(Enum):
    """Document classification for strategy adaptation."""
    RESUME = "resume"
    NARRATIVE = "narrative"
    TECHNICAL = "technical"
    UNKNOWN = "unknown"


class AnswerStyle(Enum):
    """Answer style based on question type."""
    FACT = "fact"           # Direct, concise answers
    EXPLANATORY = "explanatory"  # Detailed explanations
    SUMMARY = "summary"     # High-level overviews


@dataclass
class AdvisoryDecision:
    """
    Output of Memory Adviser containing strategy recommendations.
    
    These recommendations influence HOW the system retrieves and presents
    information, but NEVER WHAT information is retrieved.
    """
    use_memory: bool
    answer_style: str
    document_type: str
    reasoning: str  # Human-readable explanation for transparency
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "use_memory": self.use_memory,
            "answer_style": self.answer_style,
            "document_type": self.document_type,
            "reasoning": self.reasoning
        }


class MemoryAdviser:
    """
    Decides when to use conversation memory and what answer style to use.
    
    Follow-up questions need context; resume queries need different handling
    than technical docs. Only adjusts strategy, doesn't change content.
    """
    
    # Resume keywords for document type detection
    RESUME_KEYWORDS = {
        "skills", "experience", "education", "projects", "work",
        "qualifications", "certifications", "achievements",
        "professional", "career", "employment", "internship",
        "technologies", "languages", "tools", "frameworks"
    }
    
    # Technical document indicators
    TECHNICAL_KEYWORDS = {
        "api", "function", "method", "class", "module",
        "implementation", "algorithm", "data structure",
        "code", "syntax", "parameter", "return"
    }
    
    # Follow-up indicators (pronouns and references)
    FOLLOWUP_PATTERNS = [
        r"\bthis\b", r"\bthat\b", r"\bthese\b", r"\bthose\b",
        r"\bit\b", r"\bthe\b\s+\w+",  # "the document", "the project"
        r"\b(they|them|their)\b",
        r"\b(what|which)\s+(one|ones)\b",
        r"\b(above|below|mentioned|previous|earlier)\b",
        r"^\s*(and|but|also|additionally|furthermore)\s+",
    ]
    
    # Question type patterns
    FACT_PATTERNS = [
        r"^(what|which|who|when|where|how many|how much|list|name)",
        r"\b(is|are|was|were|did|does|do|can|will)\b",
    ]
    
    EXPLANATORY_PATTERNS = [
        r"^(explain|describe|how|why|what is|what are)",
        r"\b(mean|meaning|purpose|reason|cause|effect)",
    ]
    
    def __init__(self):
        """Compile regex patterns."""
        self.followup_regex = re.compile(
            "|".join(self.FOLLOWUP_PATTERNS), 
            re.IGNORECASE
        )
        self.fact_regex = re.compile(
            "|".join(self.FACT_PATTERNS), 
            re.IGNORECASE
        )
        self.explanatory_regex = re.compile(
            "|".join(self.EXPLANATORY_PATTERNS), 
            re.IGNORECASE
        )
    
    def detect_document_type(self, document_metadata: Dict[str, Any]) -> DocumentType:
        """Detect document type from keywords and filename hints."""
        # Get text content from metadata if available
        sample_text = document_metadata.get("sample_text", "").lower()
        filename = document_metadata.get("filename", "").lower()
        
        # Count keyword matches
        resume_score = sum(1 for kw in self.RESUME_KEYWORDS if kw in sample_text)
        technical_score = sum(1 for kw in self.TECHNICAL_KEYWORDS if kw in sample_text)
        
        # Filename hints
        if any(term in filename for term in ["resume", "cv", "curriculum", "profile"]):
            resume_score += 5
        
        # Classification logic
        if resume_score >= 3:
            return DocumentType.RESUME
        elif technical_score >= 3:
            return DocumentType.TECHNICAL
        elif sample_text:
            return DocumentType.NARRATIVE
        else:
            return DocumentType.UNKNOWN
    
    def detect_followup_question(
        self, 
        question: str, 
        conversation_history: Optional[List[Dict]] = None
    ) -> bool:
        """Detect if question needs conversation context (pronouns, references)."""
        # No history = no follow-up possible
        if not conversation_history or len(conversation_history) == 0:
            return False
        
        question_lower = question.lower().strip()
        
        # Check for follow-up patterns
        if self.followup_regex.search(question_lower):
            return True
        
        # Very short questions after context established are often follow-ups
        if len(question_lower.split()) <= 3 and len(conversation_history) > 0:
            return True
        
        # "More" or "continue" patterns
        if any(term in question_lower for term in ["more", "continue", "go on", "elaborate"]):
            return True
        
        return False
    
    def determine_answer_style(self, question: str, document_type: DocumentType) -> AnswerStyle:
        """Pick answer style: fact for resumes, explanatory for technical docs."""
        question_lower = question.lower().strip()
        
        # Check for explanatory patterns first
        if self.explanatory_regex.search(question_lower):
            return AnswerStyle.EXPLANATORY
        
        # Check for fact patterns
        if self.fact_regex.search(question_lower):
            # Resumes prefer facts even for "what" questions
            if document_type == DocumentType.RESUME:
                return AnswerStyle.FACT
            return AnswerStyle.FACT
        
        # Default based on document type
        if document_type == DocumentType.RESUME:
            return AnswerStyle.FACT
        elif document_type == DocumentType.TECHNICAL:
            return AnswerStyle.EXPLANATORY
        else:
            return AnswerStyle.SUMMARY
    
    def decide_strategy(
        self,
        question: str,
        document_metadata: Dict[str, Any],
        conversation_history: Optional[List[Dict]] = None
    ) -> AdvisoryDecision:
        """Main entry point: analyze query and return strategy decision."""
        # Step 1: Detect document type
        doc_type = self.detect_document_type(document_metadata)
        
        # Step 2: Detect if this is a follow-up
        is_followup = self.detect_followup_question(question, conversation_history)
        
        # Step 3: Determine answer style
        answer_style = self.determine_answer_style(question, doc_type)
        
        # Step 4: Build reasoning string
        reasons = []
        
        if is_followup:
            reasons.append("follow-up question detected")
        
        if doc_type == DocumentType.RESUME:
            reasons.append("resume-style document")
        elif doc_type == DocumentType.TECHNICAL:
            reasons.append("technical documentation")
        
        if answer_style == AnswerStyle.FACT:
            reasons.append("fact-based question")
        elif answer_style == AnswerStyle.EXPLANATORY:
            reasons.append("explanatory question")
        
        reasoning = f"Strategy: {', '.join(reasons)} if reasons else 'default strategy'"
        
        return AdvisoryDecision(
            use_memory=is_followup,
            answer_style=answer_style.value,
            document_type=doc_type.value,
            reasoning=reasoning
        )
    
    def get_prompt_modifier(self, decision: AdvisoryDecision) -> str:
        """Get prompt modifier to adjust answer style (not content)."""
        modifiers = []
        
        if decision.answer_style == "fact":
            modifiers.append("Provide a concise, factual answer.")
        elif decision.answer_style == "explanatory":
            modifiers.append("Provide a clear explanation with context.")
        elif decision.answer_style == "summary":
            modifiers.append("Provide a high-level summary.")
        
        if decision.document_type == "resume":
            modifiers.append("Focus on specific skills, experience, and qualifications.")
        
        return " ".join(modifiers) if modifiers else ""


def decide_strategy(
    question: str,
    document_metadata: Dict[str, Any],
    conversation_history: Optional[List[Dict]] = None
) -> AdvisoryDecision:
    """Convenience function for direct usage."""
    adviser = MemoryAdviser()
    return adviser.decide_strategy(question, document_metadata, conversation_history)
