# Comment Refinement Examples

Examples of how comments were simplified to sound more natural and engineer-focused.

## Before → After Examples

### Example 1: Module Docstring

**Before:**
```python
"""
Memory Adviser Module

Lightweight advisory layer that analyzes queries and document characteristics
to optimize retrieval strategy without weakening hallucination protection.

Design Principles:
1. NO new knowledge injection
2. NO override of retrieved context
3. Rule-based decisions (no ML/LLM calls)
4. Minimal complexity, maximum clarity
"""
```

**After:**
```python
"""
Memory adviser: analyzes queries to optimize retrieval strategy.

Rule-based only - no LLM calls. Doesn't inject content, just adjusts strategy.
"""
```

**Why:** Removed marketing language ("Lightweight advisory layer"), removed numbered lists that sound like documentation, kept the essential info.

---

### Example 2: Class Docstring

**Before:**
```python
class RAGChain:
    """
    RAG Pipeline implementation.
    
    DESIGN DECISION: Ollama for LLM
    - Runs locally, completely free and private
    - No API keys or rate limits
    - Supports multiple models (Llama 3.2, Mistral, etc.)
    - Easy to swap models for different needs
    
    DESIGN DECISION: Custom Prompt Template
    - Explicitly instructs LLM to ONLY use provided context
    - Reduces hallucination by constraining answer sources
    - Includes source citation instruction for transparency
    """
```

**After:**
```python
class RAGChain:
    """
    RAG pipeline using Ollama (local LLM) and ChromaDB.
    
    Using Ollama because it runs locally without API keys.
    Custom prompt constrains LLM to only use provided context.
    """
```

**Why:** Removed "DESIGN DECISION" headers (sounds like a design doc), removed bullet lists, condensed to essential reasoning.

---

### Example 3: Function Docstring

**Before:**
```python
def ask(self, question: str) -> QAResponse:
    """
    Ask a question and get an answer with sources.
    
    Flow:
    1. Validate input
    2. Retrieve relevant chunks from vector store
    3. Check if retrieved context is relevant
    4. Send to LLM with strict prompt
    5. Validate response for hallucination control
    
    Args:
        question: User's question
        
    Returns:
        QAResponse with answer and source citations
        
    Raises:
        ValueError: If vector store is empty or input is invalid
        RuntimeError: If Ollama is not running or model not found
    """
```

**After:**
```python
def ask(self, question: str) -> QAResponse:
    """Query the RAG system and return answer with sources."""
```

**Why:** Removed step-by-step flow (code shows this), removed verbose Args/Returns/Raises (type hints already show this), kept it concise.

---

### Example 4: Inline Comments

**Before:**
```python
# Step 1: Load and chunk document
loader = DocumentLoader(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
chunks = loader.load_and_chunk(document_path)

# Step 2: Initialize vector store and add documents
print("\n[2/4] Generating embeddings and storing in ChromaDB...")

# Create unique persist directory based on document name
import hashlib
doc_hash = hashlib.md5(document_path.encode()).hexdigest()[:8]
persist_dir = f"./chroma_db_{doc_hash}"
```

**After:**
```python
loader = DocumentLoader(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
chunks = loader.load_and_chunk(document_path)

print("\n[2/4] Generating embeddings and storing in ChromaDB...")

import hashlib
doc_hash = hashlib.md5(document_path.encode()).hexdigest()[:8]
persist_dir = f"./chroma_db_{doc_hash}"
```

**Why:** Removed obvious comments ("Step 1:", "Step 2:"), removed comments that just restate what the code does ("Create unique persist directory...").

---

### Example 5: "Why" Comments

**Before:**
```python
def detect_followup_question(
    self, 
    question: str, 
    conversation_history: Optional[List[Dict]] = None
) -> bool:
    """
    Detect if question relies on previous conversation context.
    
    WHY: Follow-ups like "What about this?" or "Tell me more"
    need memory to resolve pronouns and references.
    """
```

**After:**
```python
def detect_followup_question(
    self, 
    question: str, 
    conversation_history: Optional[List[Dict]] = None
) -> bool:
    """Detect if question needs conversation context (pronouns, references)."""
```

**Why:** Kept the "why" info but condensed it into the docstring. Removed the "WHY:" header which sounds tutorial-like.

---

## Principles Applied

1. **Remove obvious comments**: If the code clearly shows what's happening, don't comment it.
2. **Prefer "why" over "what"**: Comments should explain reasoning, not restate code.
3. **Avoid tutorial language**: No "Step 1:", "DESIGN DECISION:", numbered lists.
4. **Keep it short**: One-line docstrings are fine for simple functions.
5. **Remove marketing speak**: No "lightweight", "enterprise-grade", "production-ready" unless actually relevant.
