"""
CLI Entry Point with Bonus Features

Enhanced main.py with:
- Conversation memory support
- Evaluation metrics
- Chunking comparison mode
"""

import argparse
import sys
from pathlib import Path

from src.loader import DocumentLoader, validate_document_path, compare_chunking_strategies
from src.vector_store import VectorStoreManager
from src.qa_chain import RAGChain, check_ollama_status
from src.conversation_memory import EnhancedRAGChainWithMemory, ConversationMemory
from src.evaluation import RAGEvaluator
from src.memory_adviser import MemoryAdviser, decide_strategy
from src.reference_resolver import ReferenceResolver


def print_banner():
    """Print welcome banner."""
    print("\n" + "=" * 60)
    print("  Intelligent Document Q&A System (RAG) - BONUS EDITION")
    print("=" * 60)
    print("\nPowered by:")
    print("  • LangChain (RAG framework)")
    print("  • ChromaDB (Vector storage)")
    print("  • Sentence-Transformers (Embeddings)")
    print("  • Ollama (Local LLM)")
    print("\nBONUS Features:")
    print("  • Streamlit Web Interface (app.py)")
    print("  • Multi-format Support (PDF, TXT, DOCX, CSV, XLSX, MD)")
    print("  • Conversation Memory")
    print("  • Evaluation Metrics")
    print("  • Chunking Comparison")
    print("  • Unit Tests")
    print("\n" + "=" * 60)


def setup_qa_system(document_path, chunk_size=500, chunk_overlap=100, use_memory=False, auto_chunking=False):
    """Initialize QA pipeline with optional memory and auto-chunking."""
    print(f"\n[1/4] Loading document: {document_path}")
    
    # Setup loader with auto-chunking if enabled
    loader = DocumentLoader(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        auto_chunking=auto_chunking,
        filename=document_path
    )
    chunks = loader.load_and_chunk(document_path)
    
    if not chunks:
        print("Error: No content extracted")
        sys.exit(1)
    
    print(f"       ✓ Created {len(chunks)} chunks")
    
    # Extract sample text for document type detection (Memory Adviser)
    sample_text = " ".join([c.page_content[:300] for c in chunks[:3]])
    document_metadata = {
        "filename": Path(document_path).name,
        "sample_text": sample_text,
        "chunk_count": len(chunks)
    }
    
    print("\n[2/4] Storing embeddings...")
    vector_store = VectorStoreManager(persist_directory="./chroma_db")
    vector_store.add_documents(chunks)
    print(f"       ✓ Vector store ready")
    
    print("\n[3/4] Checking Ollama...")
    ollama_status = check_ollama_status()
    if ollama_status["status"] != "running":
        print("       ✗ Ollama not running!")
        sys.exit(1)
    
    model = "llama3.2"
    if model not in ollama_status["models"]:
        model = ollama_status["models"][0] if ollama_status["models"] else None
    
    print(f"       ✓ Using model: {model}")
    
    print("\n[4/4] Building RAG pipeline...")
    qa_chain = RAGChain(
        vector_store_manager=vector_store,
        model_name=model,
        temperature=0.0
    )
    
    # Wrap with memory if requested
    if use_memory:
        qa_chain = EnhancedRAGChainWithMemory(qa_chain, ConversationMemory())
        print("       ✓ RAG pipeline ready (with memory)")
    else:
        print("       ✓ RAG pipeline ready")
    
    return qa_chain, document_metadata


def run_interactive_cli(qa_chain, document_metadata, enable_eval=False):
    """Run interactive CLI with optional evaluation and Memory Adviser."""
    evaluator = RAGEvaluator() if enable_eval else None
    eval_history = []
    
    # Initialize Memory Adviser
    adviser = MemoryAdviser()
    conversation_history = []  # Track for follow-up detection
    
    print("\n" + "=" * 60)
    print("  Ready for questions!")
    if enable_eval:
        print("  📊 Evaluation mode ON")
    print("  💡 Memory Adviser active (detects follow-ups)")
    print("  Type 'exit' to quit, 'help' for commands")
    print("=" * 60)
    
    while True:
        print("\n" + "-" * 60)
        try:
            question = input("\nYour question: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nGoodbye!")
            break
        
        if not question:
            print("Please enter a question or type 'exit' to quit.")
            continue
            
        if question.lower() == "exit":
            # Show evaluation report if enabled
            if enable_eval and eval_history:
                print("\n" + evaluator.generate_report(eval_history))
            
            # Export conversation if memory is enabled
            if hasattr(qa_chain, 'export_conversation'):
                filepath = qa_chain.export_conversation()
                print(f"\n💾 Conversation saved to: {filepath}")
            
            print("\nGoodbye!")
            break
            
        if question.lower() == "help":
            print("\nCommands:")
            print("  exit      - Quit the program")
            print("  help      - Show this help")
            print("  stats     - Show conversation stats (if memory enabled)")
            print("  export    - Export conversation (if memory enabled)")
            print("  advise    - Show Memory Adviser status")
            print("\nMemory Adviser Tips:")
            print("  - Ask follow-ups with: 'this', 'that', 'more', 'continue'")
            print("  - First questions ignore memory (fresh context)")
            print("  - Resume docs trigger fact-based answers automatically")
            continue
        
        if question.lower() == "stats" and hasattr(qa_chain, 'get_conversation_stats'):
            stats = qa_chain.get_conversation_stats()
            print("\n📊 Conversation Stats:")
            for key, value in stats.items():
                print(f"  {key}: {value}")
            continue
        
        if question.lower() == "export" and hasattr(qa_chain, 'export_conversation'):
            filepath = qa_chain.export_conversation()
            print(f"\n💾 Conversation exported to: {filepath}")
            continue
        
        if question.lower() == "advise":
            """Show Memory Adviser debug info for last query."""
            print("\n🧠 Memory Adviser Status:")
            print(f"  Document type: {document_metadata.get('filename', 'unknown')}")
            print(f"  Conversation turns: {len(conversation_history)}")
            print("  Hint: Ask a follow-up question with 'this', 'that', or 'more'")
            continue
        
        if len(question) > 1000:
            print("Error: Question too long (max 1000 chars)")
            continue
        
        # MEMORY ADVISER: Analyze question before retrieval
        decision = adviser.decide_strategy(
            question=question,
            document_metadata=document_metadata,
            conversation_history=conversation_history if conversation_history else None
        )
        
        # Show advisory decision (transparent to user)
        print(f"\n🎯 Strategy: {decision.answer_style.upper()} | Memory: {'ON' if decision.use_memory else 'OFF'} | Doc: {decision.document_type.upper()}")
        
        # REFERENCE RESOLVER: Expand pronouns using previous answer
        original_question = question
        if decision.use_memory and conversation_history:
            resolver = ReferenceResolver(debug=True)
            resolved_question = resolver.resolve(question, qa_chain.memory)
            if resolved_question != question:
                print(f"🔁 Expanded: {resolved_question[:100]}...")
                question = resolved_question
        
        # Process question
        try:
            print("\nThinking...")
            
            import time
            start_time = time.time()
            
            # Ask question with memory if enabled and decision allows
            if hasattr(qa_chain, 'ask'):
                response = qa_chain.ask(question, memory_decision=decision)
            else:
                response = qa_chain.ask(question)
            
            latency = (time.time() - start_time) * 1000
            
            # Store in conversation history for follow-up detection
            conversation_history.append({
                "question": question,
                "answer": response.answer,
                "sources": len(response.sources)
            })
            
            # Display response
            formatted = qa_chain.format_response_with_sources(response) if hasattr(qa_chain, 'format_response_with_sources') else response.answer
            print(formatted)
            
            # Evaluation metrics
            if enable_eval and evaluator:
                quality = evaluator.evaluate_answer_quality(response.answer, response.sources)
                print(f"\n📊 Quality Score: {quality['grounding_ratio']:.2%} grounded in sources")
                
        except Exception as e:
            print(f"\nError: {e}")


def run_chunking_comparison(document_path):
    """BONUS: Run chunking strategy comparison."""
    print("\n" + "=" * 60)
    print("  CHUNKING STRATEGY COMPARISON")
    print("=" * 60)
    
    strategies = [
        {"name": "recursive", "chunk_size": 300, "chunk_overlap": 50, "type": "recursive"},
        {"name": "recursive", "chunk_size": 500, "chunk_overlap": 100, "type": "recursive"},
        {"name": "recursive", "chunk_size": 800, "chunk_overlap": 150, "type": "recursive"},
    ]
    
    print(f"\nComparing {len(strategies)} configurations...")
    print(f"Document: {document_path}\n")
    
    results = compare_chunking_strategies(document_path, strategies=[s["type"] for s in strategies])
    
    print("RESULTS:")
    print("-" * 60)
    for key, result in results.items():
        if "error" in result:
            print(f"❌ {key}: ERROR - {result['error']}")
        else:
            print(f"\n✓ {key}:")
            print(f"  Chunks: {result['num_chunks']}")
            print(f"  Avg Length: {result['avg_chunk_length']:.0f} chars")
            print(f"  Sample: {result['sample_chunk'][:80]}...")
    
    print("\n" + "=" * 60)


def main():
    """Main entry point with all bonus features."""
    parser = argparse.ArgumentParser(
        description="Intelligent Document Q&A System - BONUS EDITION",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main_bonus.py --document data/sample.pdf
  python main_bonus.py --document data/resume.pdf --auto-chunk
  python main_bonus.py --document data/manual.pdf --auto-chunk --memory
  python main_bonus.py --document data/report.docx --chunk-size 300 --chunk-overlap 50
  python main_bonus.py --compare-chunking data/sample.txt
  streamlit run app.py  (for web interface)
        """
    )
    
    parser.add_argument("--document", "-d", help="Path to document")
    parser.add_argument("--reset", "-r", action="store_true", help="Clear vector store")
    parser.add_argument("--chunk-size", type=int, default=500, help="Manual chunk size (ignored if --auto-chunk)")
    parser.add_argument("--chunk-overlap", type=int, default=100, help="Manual chunk overlap (ignored if --auto-chunk)")
    parser.add_argument("--auto-chunk", "-a", action="store_true", help="Auto-select chunking strategy based on document type")
    parser.add_argument("--memory", "-m", action="store_true", help="Enable conversation memory")
    parser.add_argument("--eval", "-e", action="store_true", help="Enable evaluation metrics")
    parser.add_argument("--compare-chunking", help="Run chunking comparison on document")
    
    args = parser.parse_args()
    
    # Chunking comparison mode
    if args.compare_chunking:
        if not validate_document_path(args.compare_chunking):
            sys.exit(1)
        run_chunking_comparison(args.compare_chunking)
        return
    
    # Normal Q&A mode
    if not args.document:
        parser.print_help()
        sys.exit(1)
    
    if not validate_document_path(args.document):
        sys.exit(1)
    
    # Validate parameters
    if args.chunk_size < 100 or args.chunk_size > 2000:
        print("Error: Chunk size must be 100-2000")
        sys.exit(1)
    
    print_banner()
    
    if args.reset:
        print("\n[!] Clearing vector store...")
        VectorStoreManager().clear_store()
    
    try:
        qa_chain, document_metadata = setup_qa_system(
            args.document,
            args.chunk_size,
            args.chunk_overlap,
            use_memory=args.memory,
            auto_chunking=args.auto_chunk
        )
    except Exception as e:
        print(f"\nSetup error: {e}")
        sys.exit(1)
    
    try:
        run_interactive_cli(qa_chain, document_metadata, enable_eval=args.eval)
    except KeyboardInterrupt:
        print("\n\nInterrupted. Goodbye!")


if __name__ == "__main__":
    main()
