"""
Reference Resolver Module

Lightweight, rule-based pronoun resolution for conversational RAG.
Expands follow-up questions using previous answer content BEFORE retrieval.

Design Principles:
1. Rule-based ONLY (no ML, no LLM calls)
2. Expands ONLY from previous answers (no new knowledge)
3. Deterministic and explainable
4. Safe - does NOT affect grounding validation
5. Minimal, readable implementation

When to Apply:
- Memory Adviser detects use_memory = True
- Question contains pronouns: "this", "that", "these", "those", "above", "previous"
"""

from typing import Optional, List, Dict, Any
import re


def extract_key_items(answer: str, max_items: int = 5) -> List[str]:
    """
    Extract key items/entities from the previous answer.
    
    Uses multiple strategies:
    1. Bullet-like patterns (lines starting with -, *, •)
    2. Numbered lists (1., 2., etc.)
    3. Comma-separated lists in the answer
    4. Key phrases (noun phrases after verbs like "includes", "are")
    
    Args:
        answer: The previous assistant answer text
        max_items: Maximum items to extract
        
    Returns:
        List of key item strings
    """
    items = []
    
    # Strategy 1: Bullet-like patterns
    bullet_pattern = r'^[\s]*[-*•][\s]+(.+)$'
    bullet_matches = re.findall(bullet_pattern, answer, re.MULTILINE)
    items.extend([m.strip() for m in bullet_matches[:max_items]])
    
    # Strategy 2: Numbered lists
    if not items:
        numbered_pattern = r'^[\s]*\d+[.)][\s]+(.+)$'
        numbered_matches = re.findall(numbered_pattern, answer, re.MULTILINE)
        items.extend([m.strip() for m in numbered_matches[:max_items]])
    
    # Strategy 3: Look for patterns like "X, Y, and Z" or "includes A, B, C"
    if not items:
        # Pattern: "includes/are: item1, item2, item3"
        list_pattern = r'(?:includes?|are|consists? of)[\s:]+([^,.]+(?:,[^,.]+){1,})'
        list_match = re.search(list_pattern, answer, re.IGNORECASE)
        if list_match:
            # Split by commas and "and"
            list_text = list_match.group(1)
            split_items = re.split(r',|\band\b', list_text)
            items.extend([item.strip() for item in split_items[:max_items] if item.strip()])
    
    # Strategy 4: Extract noun phrases after key verbs
    if not items:
        # Look for "X provides Y", "X supports Y", "X includes Y"
        verb_pattern = r'(?:provides?|supports?|includes?|offers?|contains?)[\s:]+([^,.]{5,50})'
        verb_matches = re.findall(verb_pattern, answer, re.IGNORECASE)
        items.extend([m.strip() for m in verb_matches[:max_items]])
    
    # Strategy 5: Fallback - split by newlines and take substantial lines
    if not items:
        lines = [line.strip() for line in answer.split('\n') if len(line.strip()) > 10]
        items.extend(lines[:max_items])
    
    return items[:max_items]


def contains_pronouns(question: str) -> bool:
    """
    Check if question contains reference pronouns.
    
    Args:
        question: User's question
        
    Returns:
        True if pronouns detected
    """
    pronouns = ['this', 'that', 'these', 'those', 'above', 'previous', 'mentioned', 'it']
    question_lower = question.lower()
    
    # Check for standalone pronouns (not part of other words)
    for pronoun in pronouns:
        # Use word boundaries to avoid matching "thesis" as "this"
        pattern = r'\b' + pronoun + r'\b'
        if re.search(pattern, question_lower):
            return True
    
    return False


def resolve_references(
    question: str, 
    conversation_memory: Any,
    debug: bool = False
) -> str:
    """
    Resolve pronoun references in follow-up questions.
    
    Expands questions like "Which of these are related to AI?" into:
    "Which of the following are related to AI: item1, item2, item3?"
    
    Args:
        question: Current user question
        conversation_memory: ConversationMemory object with history
        debug: Whether to print debug info
        
    Returns:
        Expanded question (or original if no resolution needed)
    """
    # Check if question contains pronouns
    if not contains_pronouns(question):
        return question
    
    # Get last assistant answer from memory
    if not conversation_memory or not hasattr(conversation_memory, 'history'):
        return question
    
    history = conversation_memory.history
    if not history:
        return question
    
    # Get the last assistant answer
    last_turn = history[-1]
    last_answer = last_turn.answer if hasattr(last_turn, 'answer') else str(last_turn)
    
    if not last_answer or len(last_answer) < 10:
        return question
    
    # Extract key items from the answer
    key_items = extract_key_items(last_answer, max_items=5)
    
    if not key_items:
        return question
    
    # Rewrite the question with resolved references
    items_text = ", ".join(key_items)
    
    # Determine the best joining phrase based on question structure
    question_lower = question.lower()
    
    if 'which of' in question_lower or 'which one' in question_lower:
        # "Which of these..." -> "Which of the following..."
        expanded = f"Based on these items: {items_text}. {question}"
    elif 'what about' in question_lower or 'how about' in question_lower:
        # "What about this?" -> "What about [specific item]?"
        expanded = f"Regarding {items_text}. {question}"
    elif 'are these' in question_lower or 'is this' in question_lower:
        # "Are these related to X?" -> "Are [items] related to X?"
        expanded = f"Regarding: {items_text}. {question}"
    else:
        # Default expansion
        expanded = f"Context: {items_text}. {question}"
    
    if debug:
        print(f"🔁 Expanded Question: {expanded}")
    
    return expanded


def get_resolution_summary(
    original_question: str,
    expanded_question: str
) -> Optional[Dict[str, Any]]:
    """
    Get a summary of what was resolved (for transparency).
    
    Returns:
        Dict with resolution info or None if no resolution
    """
    if original_question == expanded_question:
        return None
    
    return {
        "original": original_question,
        "expanded": expanded_question,
        "was_resolved": True,
        "reason": "Pronoun reference resolution applied"
    }


class ReferenceResolver:
    """
    Wrapper class for reference resolution functionality.
    Provides a clean interface for integration.
    """
    
    def __init__(self, debug: bool = False):
        """
        Initialize the resolver.
        
        Args:
            debug: Enable debug output
        """
        self.debug = debug
    
    def resolve(
        self, 
        question: str, 
        conversation_memory: Any
    ) -> str:
        """
        Resolve references in a question.
        
        Args:
            question: User question
            conversation_memory: Conversation memory object
            
        Returns:
            Resolved question
        """
        return resolve_references(question, conversation_memory, self.debug)
    
    def should_resolve(self, question: str) -> bool:
        """
        Check if a question needs reference resolution.
        
        Args:
            question: User question
            
        Returns:
            True if pronouns detected
        """
        return contains_pronouns(question)
