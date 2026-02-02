"""
Conversation memory for multi-turn Q&A sessions.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import json
from pathlib import Path


@dataclass
class ConversationTurn:
    """Represents a single turn in the conversation."""
    question: str
    answer: str
    sources: List[Dict[str, Any]] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "sources": self.sources,
            "timestamp": self.timestamp.isoformat()
        }


class ConversationMemory:
    """Stores conversation history for context-aware follow-up questions."""
    
    def __init__(self, max_history: int = 10, session_id: Optional[str] = None):
        """
        Initialize conversation memory.
        
        Args:
            max_history: Maximum number of turns to remember
            session_id: Unique identifier for this conversation session
        """
        self.max_history = max_history
        self.session_id = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.history: List[ConversationTurn] = []
        self.metadata: Dict[str, Any] = {
            "session_id": self.session_id,
            "created_at": datetime.now().isoformat(),
            "document": None
        }
    
    def add_turn(self, question: str, answer: str, sources: List[Dict] = None):
        """
        Add a new conversation turn.
        
        Args:
            question: User's question
            answer: System's answer
            sources: Source documents used
        """
        turn = ConversationTurn(
            question=question,
            answer=answer,
            sources=sources or []
        )
        
        self.history.append(turn)
        
        # Maintain max history limit
        if len(self.history) > self.max_history:
            self.history.pop(0)
    
    def get_context(self, window: int = 3) -> str:
        """
        Get recent conversation context for prompt augmentation.
        
        BONUS: Enables follow-up questions by providing conversation context.
        
        Args:
            window: Number of recent turns to include
            
        Returns:
            Formatted conversation history string
        """
        if not self.history:
            return ""
        
        recent = self.history[-window:]
        context_parts = []
        
        for turn in recent:
            context_parts.append(f"Q: {turn.question}")
            context_parts.append(f"A: {turn.answer}")
        
        return "\n".join(context_parts)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get conversation statistics.
        
        BONUS: Evaluation metrics for conversation quality.
        """
        if not self.history:
            return {
                "total_turns": 0,
                "avg_question_length": 0,
                "avg_answer_length": 0,
                "sessions_with_sources": 0
            }
        
        total_turns = len(self.history)
        avg_q_len = sum(len(t.question) for t in self.history) / total_turns
        avg_a_len = sum(len(t.answer) for t in self.history) / total_turns
        with_sources = sum(1 for t in self.history if t.sources)
        
        return {
            "total_turns": total_turns,
            "avg_question_length": round(avg_q_len, 2),
            "avg_answer_length": round(avg_a_len, 2),
            "sessions_with_sources": with_sources,
            "source_citation_rate": round(with_sources / total_turns * 100, 2)
        }
    
    def export_to_json(self, filepath: Optional[str] = None) -> str:
        """
        Export conversation to JSON file.
        
        Args:
            filepath: Output file path (optional)
            
        Returns:
            Path to exported file
        """
        data = {
            "metadata": self.metadata,
            "stats": self.get_stats(),
            "conversation": [turn.to_dict() for turn in self.history]
        }
        
        if filepath is None:
            filepath = f"conversation_{self.session_id}.json"
        
        Path(filepath).write_text(json.dumps(data, indent=2))
        return filepath
    
    def clear(self):
        """Clear conversation history."""
        self.history.clear()
    
    def __len__(self):
        return len(self.history)


class EnhancedRAGChainWithMemory:
    """
    BONUS: RAG Chain with conversation memory integration.
    
    Wraps the base RAGChain and adds memory capabilities.
    """
    
    def __init__(self, base_qa_chain, memory: Optional[ConversationMemory] = None):
        """
        Initialize with base QA chain and optional memory.
        
        Args:
            base_qa_chain: The underlying RAGChain instance
            memory: ConversationMemory instance (creates new if None)
        """
        self.qa_chain = base_qa_chain
        self.memory = memory or ConversationMemory()
    
    def ask(self, question: str, memory_decision: Optional[Any] = None) -> Any:
        """
        Ask a question with optional memory context.
        
        Args:
            question: User's question
            memory_decision: Memory Adviser decision object
            
        Returns:
            QAResponse with answer and sources
        """
        # Use Memory Adviser decision if provided, otherwise default to True
        use_memory = False
        if memory_decision:
            use_memory = memory_decision.use_memory
        elif len(self.memory) > 0:
            # Fallback: use memory if there's history and no adviser
            use_memory = True
        
        # Enhance question with context if memory is enabled
        if use_memory and len(self.memory) > 0:
            context = self.memory.get_context()
            enhanced_prompt = f"""Previous conversation context:
{context}

Current question: {question}

Please answer the current question, taking into account the previous conversation if relevant."""
        else:
            enhanced_prompt = question
        
        # Get response from base chain
        response = self.qa_chain.ask(enhanced_prompt)
        
        # Store in memory
        sources = [
            {
                "file": doc.metadata.get("source_file", "Unknown"),
                "page": doc.metadata.get("page", "N/A"),
                "preview": doc.page_content[:100]
            }
            for doc in response.sources
        ]
        
        self.memory.add_turn(question, response.answer, sources)
        
        return response
    
    def get_conversation_stats(self) -> Dict[str, Any]:
        """Get conversation statistics."""
        return self.memory.get_stats()
    
    def export_conversation(self, filepath: Optional[str] = None) -> str:
        """Export conversation history."""
        return self.memory.export_to_json(filepath)
