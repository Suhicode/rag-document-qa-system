"""
CLI entry point for document Q&A system.

Usage:
    python main.py --document data/sample.pdf
"""

import argparse
import sys
from pathlib import Path

from loader import DocumentLoader, validate_document_path
from vector_store import VectorStoreManager
from qa_chain import RAGChain, check_ollama_status


def print_banner():
    """Print startup banner."""
    print("\n" + "=" * 60)
    print("  Intelligent Document Q&A System (RAG)")
    print("=" * 60)
    print("\nPowered by:")
    print("  • LangChain (RAG framework)")
    print("  • ChromaDB (Vector storage)")
    print("  • Sentence-Transformers (Embeddings)")
    print("  • Ollama (Local LLM)")
    print("\n" + "=" * 60)


def setup_qa_system(document_path: str, chunk_size: int = 500, chunk_overlap: int = 100) -> RAGChain:
    """Initialize QA pipeline: load doc, chunk, embed, build RAG chain."""
    print(f"\n[1/4] Loading document: {document_path}")
    
    loader = DocumentLoader(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = loader.load_and_chunk(document_path)
    
    if not chunks:
        print("Error: No content could be extracted from document")
        sys.exit(1)
    
    print(f"       ✓ Created {len(chunks)} chunks (size: {chunk_size}, overlap: {chunk_overlap})")
    
    if len(chunks) > 1000:
        print(f"       ⚠ Warning: Large document ({len(chunks)} chunks). Consider increasing chunk size.")
    
    print("\n[2/4] Generating embeddings and storing in ChromaDB...")
    
    import hashlib
    doc_hash = hashlib.md5(document_path.encode()).hexdigest()[:8]
    persist_dir = f"./chroma_db_{doc_hash}"
    
    vector_store = VectorStoreManager(persist_directory=persist_dir)
    vector_store.add_documents(chunks)
    
    stats = vector_store.get_collection_stats()
    print(f"       ✓ Vector store ready ({stats['document_count']} documents)")
    
    print("\n[3/4] Checking Ollama status...")
    ollama_status = check_ollama_status()
    
    if ollama_status["status"] != "running":
        print("       ✗ Ollama is not running!")
        print("\nPlease ensure:")
        print("  1. Install Ollama: https://ollama.com/download")
        print("  2. Start Ollama server")
        print("  3. Pull a model: ollama pull llama3.2")
        sys.exit(1)
    
    print(f"       ✓ Ollama running with models: {ollama_status['models']}")
    
    print("\n[4/4] Building RAG pipeline...")
    
    model = "llama3.2"
    if model not in ollama_status["models"]:
        alternatives = ["mistral", "llama2", "phi"]
        for alt in alternatives:
            if alt in ollama_status["models"]:
                model = alt
                break
        else:
            model = ollama_status["models"][0] if ollama_status["models"] else None
            if not model:
                print("       ✗ No models found. Run: ollama pull llama3.2")
                sys.exit(1)
    
    print(f"       Using model: {model}")
    
    qa_chain = RAGChain(
        vector_store_manager=vector_store,
        model_name=model,
        temperature=0.0
    )
    
    print("       ✓ RAG pipeline ready")
    
    return qa_chain


def run_interactive_cli(qa_chain: RAGChain):
    """Run interactive Q&A loop. Type 'exit' to quit."""
    print("\n" + "=" * 60)
    print("  Ready for questions!")
    print("  Type 'exit' to quit, 'help' for commands")
    print("=" * 60)
    
    while True:
        print("\n" + "-" * 60)
        try:
            question = input("\nYour question: ").strip()
        except KeyboardInterrupt:
            print("\n\nInterrupted. Goodbye!")
            break
        except EOFError:
            print("\n\nGoodbye!")
            break
        
        # Handle special commands
        if not question:
            print("Please enter a question or type 'exit' to quit.")
            continue
            
        if question.lower() == "exit":
            print("\nGoodbye!")
            break
            
        if question.lower() == "help":
            print("\nCommands:")
            print("  exit  - Quit the program")
            print("  help  - Show this help message")
            print("\nTips:")
            print("  • Ask specific questions about document content")
            print("  • The system answers ONLY from the document")
            print("  • Source citations are provided with each answer")
            print("  • If no information is found, you'll be notified clearly")
            continue
        
        # Validate question length
        if len(question) > 1000:
            print("Error: Question too long. Please keep questions under 1000 characters.")
            continue
        
        # Process question
        try:
            print("\nThinking...")
            response = qa_chain.ask(question)
            
            # Format and display response
            formatted = qa_chain.format_response_with_sources(response)
            print(formatted)
            
        except ValueError as e:
            print(f"\nInput Error: {e}")
            print("Please try again with a different question.")
        except RuntimeError as e:
            print(f"\nSystem Error: {e}")
            print("Please check your Ollama setup and try again.")
        except Exception as e:
            print(f"\nUnexpected Error: {e}")
            print("Please try again or type 'exit' to quit.")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Intelligent Document Q&A System using RAG",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --document data/sample.pdf
  python main.py --document data/notes.txt
  python main.py --document data/manual.pdf --reset
        """
    )
    
    parser.add_argument(
        "--document",
        "-d",
        required=True,
        help="Path to document (PDF or TXT)"
    )
    
    parser.add_argument(
        "--reset",
        "-r",
        action="store_true",
        help="Clear existing vector store before loading"
    )
    
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500,
        help="Chunk size for document splitting (default: 500)"
    )
    
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=100,
        help="Chunk overlap for document splitting (default: 100)"
    )
    
    args = parser.parse_args()
    
    # Validate chunk parameters
    if args.chunk_size < 100 or args.chunk_size > 2000:
        print("Error: Chunk size must be between 100 and 2000 characters.")
        sys.exit(1)
    
    if args.chunk_overlap < 0 or args.chunk_overlap >= args.chunk_size:
        print("Error: Chunk overlap must be between 0 and chunk_size.")
        sys.exit(1)
    
    # Validate document path
    if not validate_document_path(args.document):
        sys.exit(1)
    
    # Print banner
    print_banner()
    
    # Optional: reset vector store
    if args.reset:
        print("\n[!] Clearing existing vector store...")
        temp_vs = VectorStoreManager()
        temp_vs.clear_store()
    
    # Setup QA system
    try:
        qa_chain = setup_qa_system(args.document, args.chunk_size, args.chunk_overlap)
    except KeyboardInterrupt:
        print("\n\nSetup interrupted. Exiting.")
        sys.exit(0)
    except Exception as e:
        print(f"\nError during setup: {e}")
        sys.exit(1)
    
    # Run interactive CLI
    try:
        run_interactive_cli(qa_chain)
    except KeyboardInterrupt:
        print("\n\nInterrupted. Goodbye!")
        sys.exit(0)


if __name__ == "__main__":
    main()
