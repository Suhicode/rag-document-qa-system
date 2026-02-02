# Document Q&A System

A RAG-based document question-answering system I built to solve the hallucination problem in LLM-powered document Q&A. It retrieves relevant passages from documents and uses them to ground LLM responses, ensuring answers are traceable and verifiable.

## What I Built

I needed a system that could answer questions about documents without making things up. The core idea is simple: find relevant chunks from the document, feed them to an LLM with strict instructions to only use that context, and cite sources for every answer.

The system supports multiple document formats (PDF, DOCX, TXT, CSV, etc.), maintains conversation context, and includes some optimizations I added along the way for better retrieval quality.

## Architecture

```
Document → Chunking → Embeddings → ChromaDB
                                    ↓
User Query → Memory → Embed Query → Similarity Search → LLM → Answer + Sources
```

**Tech choices:**
- **Ollama + Llama 3.2**: Runs locally, no API keys, free
- **ChromaDB**: Persistent vector storage, simple to use
- **sentence-transformers (all-MiniLM-L6-v2)**: Fast 384-dim embeddings
- **LangChain**: Handles the RAG orchestration

## Features

### Core Requirements
- PDF and TXT document loading
- Semantic chunking (configurable size/overlap)
- Vector embeddings and ChromaDB storage
- Similarity search (top-k=3)
- LLM answers constrained to document context
- Source citations with every answer
- CLI interface
- Error handling and validation

### Additional Features

**Multi-format support**: Extended beyond PDF/TXT to handle DOCX, PPTX, CSV, XLSX, MD, HTML. I added these because I kept getting different file types in practice.

**Conversation memory**: Tracks context across multiple questions. Useful for follow-ups like "Tell me more about that" or "What about his experience?"

**Chunking adviser**: Auto-detects document type (resume vs technical doc vs narrative) and picks chunk size/overlap accordingly. Resumes work better with smaller chunks (200 chars), narratives need larger ones (800 chars) to preserve flow.

**Memory adviser**: Analyzes queries to decide when to use conversation history. Follow-up questions with pronouns ("What about this?") need memory; standalone questions don't.

**Evaluation metrics**: Added relevance scoring and grounding checks to measure retrieval quality. Helpful for debugging why certain queries don't work well.

**Web interface**: Streamlit UI for easier interaction. Drag-and-drop upload, chat interface, source previews.

## Quick Start

### Setup

```bash
# Install Ollama (if not already installed)
# Windows: Download from ollama.com
# Mac/Linux: curl -fsSL https://ollama.com/install.sh | sh

# Pull a model
ollama pull llama3.2

# Install Python dependencies
pip install -r requirements.txt
```

### Usage

**CLI (basic):**
```bash
python src/main.py --document data/sample.txt
```

**CLI (with all features):**
```bash
python main_bonus.py --document data/report.docx --auto-chunk --memory --eval
```

**Web interface:**
```bash
streamlit run app.py
```

**Run tests:**
```bash
python tests/test_rag_system.py
```

## Project Structure

```
├── app.py                      # Streamlit web interface
├── main_bonus.py               # Enhanced CLI with all features
├── src/
│   ├── loader.py               # Document loading + chunking
│   ├── vector_store.py         # ChromaDB management
│   ├── qa_chain.py             # RAG pipeline
│   ├── conversation_memory.py  # Conversation history
│   ├── memory_adviser.py       # Query strategy optimization
│   ├── chunking_adviser.py     # Auto-chunking logic
│   ├── evaluation.py           # Quality metrics
│   └── main.py                 # Basic CLI
├── tests/
│   └── test_rag_system.py      # Unit tests
└── requirements.txt
```

## Development Notes

### Problems I Faced

**PDF extraction issues**: Some PDFs extract text with weird spacing (e.g., "e n v i r o n m e n t s"). I added a normalization function to rejoin character-spaced words.

**Chunking quality**: Default chunk size (500) worked poorly for resumes - they got split mid-section. I built the chunking adviser to detect document type and adjust accordingly.

**Follow-up questions**: Questions like "What about this?" failed without conversation context. Added memory tracking and reference resolution.

**Hallucination control**: Even with strict prompts, the LLM sometimes made up details. I added grounding validation that checks if answer words appear in retrieved sources (30% threshold).

### Design Decisions

**Why Ollama?** I wanted something that runs locally without API keys or rate limits. Ollama is simple to set up and works offline.

**Why ChromaDB?** It's lightweight, persistent, and doesn't require a separate server. Good enough for single-user scenarios.

**Why top-k=3?** More chunks = more noise. Three chunks usually contain enough context without overwhelming the LLM prompt.

**Why temperature=0.0?** For factual Q&A, I want deterministic, consistent answers. No creativity needed here.

**Why rule-based advisers?** I tried using LLMs to decide chunking/memory strategy, but it added latency and complexity. Simple regex-based detection works fine and is faster.

### Known Limitations

- Large documents (>1000 chunks) can be slow. Consider increasing chunk size for very long docs.
- The grounding check (30% word overlap) is heuristic-based. Sometimes legitimate answers get flagged if they paraphrase heavily.
- Conversation memory doesn't handle very long histories well - it just truncates after 10 turns.
- Multi-document support is basic - each document gets its own vector store. No cross-document search yet.

## How It Works

1. **Document loading**: Extracts text from various formats, normalizes spacing issues
2. **Chunking**: Splits document into overlapping chunks. Size/overlap auto-selected based on document type
3. **Embedding**: Converts chunks to 384-dim vectors using sentence-transformers
4. **Storage**: Stores vectors in ChromaDB with metadata (source file, page number)
5. **Query**: User asks question → embed query → similarity search → retrieve top-3 chunks
6. **Generation**: LLM generates answer using only retrieved chunks, with strict prompt to prevent hallucination
7. **Validation**: Check that answer words appear in sources (grounding check)
8. **Response**: Return answer + source citations

## Hallucination Prevention

Five layers of protection:

1. **Strict prompt**: LLM explicitly instructed to only use provided context
2. **Similarity threshold**: Only retrieve chunks above relevance threshold
3. **Grounding validation**: Verify 30% of answer words appear in sources
4. **Explicit rejection**: LLM trained to say "no information found" when context lacks answer
5. **Source citations**: Every answer includes references for verification

The advisers (memory, chunking) don't inject content - they only optimize retrieval strategy.

## Testing

Run the test suite:
```bash
python tests/test_rag_system.py
```

Tests cover:
- Document loading (all formats)
- Conversation memory
- Evaluation metrics
- QA chain functionality
- Integration tests

## License

yousuf suhail (yousufsuhaily@gmail.com)

* OUTPUT IMAGES ARE IN OUTPUT_SCREENSHOT FOLDER **
