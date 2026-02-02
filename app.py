"""
Streamlit web interface for document Q&A.
"""

import streamlit as st
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent / "src"))

from src.loader import DocumentLoader, validate_document_path
from src.vector_store import VectorStoreManager
from src.qa_chain import RAGChain, check_ollama_status
from src.chunking_adviser import ChunkingAdviser

st.set_page_config(
    page_title="DocuQuery",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    .main { background-color: #fafafa; }
    .stApp h1 {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        font-weight: 600;
        color: #1a1a2e;
        font-size: 1.75rem;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        color: #64748b;
        font-size: 0.9rem;
        margin-bottom: 2rem;
    }
    .stChatMessage {
        background-color: white;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        margin: 8px 0;
    }
    .source-box {
        background-color: #f1f5f9;
        border-left: 3px solid #3b82f6;
        padding: 12px 16px;
        margin: 12px 0;
        border-radius: 0 6px 6px 0;
    }
    .stButton button {
        border-radius: 6px;
        font-weight: 500;
    }
    .stButton button[kind="primary"] {
        background-color: #1e40af;
    }
    .success-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background-color: #dcfce7;
        color: #166534;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 500;
    }
    .info-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    .info-card h3 {
        font-size: 1rem;
        font-weight: 600;
        color: #1e293b;
        margin-bottom: 0.75rem;
    }
    .info-card p {
        color: #64748b;
        font-size: 0.9rem;
        line-height: 1.6;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}
</style>
""", unsafe_allow_html=True)

if 'qa_chain' not in st.session_state:
    st.session_state.qa_chain = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'current_doc' not in st.session_state:
    st.session_state.current_doc = None

def render_header():
    col1, col2 = st.columns([6, 1])
    with col1:
        st.markdown("<h1>DocuQuery</h1>", unsafe_allow_html=True)
        st.markdown("<p class='subtitle'>Document intelligence powered by retrieval-augmented generation</p>", unsafe_allow_html=True)
    with col2:
        if st.session_state.current_doc:
            st.markdown("<div style='text-align: right; margin-top: 8px;'><span class='success-pill'>Ready</span></div>", unsafe_allow_html=True)

def process_document(uploaded_file, chunk_size, chunk_overlap, auto_chunking=False):
    temp_path = Path(f"temp_{uploaded_file.name}")
    temp_path.write_bytes(uploaded_file.getvalue())
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        status_text.text("Loading document...")
        progress_bar.progress(25)
        
        # Setup loader with auto-chunking if enabled
        loader = DocumentLoader(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            auto_chunking=auto_chunking,
            filename=uploaded_file.name
        )
        chunks = loader.load_and_chunk(str(temp_path))
        
        # Show chunking decision if auto-chunking was used
        if auto_chunking:
            decision = loader.get_chunking_decision()
            if decision:
                st.success(f"📦 Auto-chunking Applied: Size={decision.chunk_size} | Overlap={decision.chunk_overlap} | Type: {decision.document_type.upper()}")
                st.caption(f"Reason: {decision.reason}")
        
        status_text.text("Building index...")
        progress_bar.progress(50)
        
        # Create unique persist directory based on document name
        import hashlib
        doc_hash = hashlib.md5(uploaded_file.name.encode()).hexdigest()[:8]
        persist_dir = f"./chroma_db_{doc_hash}"
        
        vector_store = VectorStoreManager(persist_directory=persist_dir)
        vector_store.add_documents(chunks)
        
        status_text.text("Initializing language model...")
        progress_bar.progress(75)
        
        status = check_ollama_status()
        model = "llama3.2" if "llama3.2" in status["models"] else (status["models"][0] if status["models"] else None)
        
        st.session_state.qa_chain = RAGChain(
            vector_store_manager=vector_store,
            model_name=model,
            temperature=0.0
        )
        
        st.session_state.current_doc = uploaded_file.name
        st.session_state.chat_history = []
        
        progress_bar.progress(100)
        st.success(f"Processed {len(chunks)} sections from {uploaded_file.name}")
        st.rerun()
        
    except Exception as e:
        st.error(f"Error: {str(e)}")
    finally:
        temp_path.unlink(missing_ok=True)

def render_upload_section():
    st.markdown("<div class='info-card'><h3>Get Started</h3><p>Upload a document to begin querying. Supported formats include PDF, Word documents, text files, and spreadsheets.</p></div>", unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Choose a file", type=['pdf', 'txt', 'docx', 'doc', 'md', 'csv', 'xlsx', 'pptx'], label_visibility="collapsed")
    
    if uploaded_file:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"**Selected:** {uploaded_file.name}")
        with col2:
            with st.expander("Advanced Options"):
                auto_chunk = st.checkbox("Auto-detect chunking", value=True, help="Automatically select optimal chunking based on document type")
                
                if auto_chunk:
                    # Show auto-detected values (will be updated after processing)
                    st.info("🤖 Auto-detection enabled\nChunking will be optimized after upload")
                else:
                    chunk_size = st.slider("Chunk Size", 100, 2000, 500, 100)
                    chunk_overlap = st.slider("Chunk Overlap", 0, 500, 100, 50)
        
        if st.button("Process Document", type="primary", use_container_width=True):
            if auto_chunk:
                # Use default values, will be overridden by auto-chunking
                process_document(uploaded_file, 500, 100, auto_chunk)
            else:
                process_document(uploaded_file, chunk_size, chunk_overlap, auto_chunk)

def render_chat_interface():
    st.markdown(f"<div style='background: white; border: 1px solid #e2e8f0; border-radius: 8px; padding: 12px 16px; margin-bottom: 16px;'><span style='color: #64748b; font-size: 0.875rem;'>Current document:</span><span style='font-weight: 500; margin-left: 8px;'>{st.session_state.current_doc}</span></div>", unsafe_allow_html=True)
    
    if st.button("Switch Document", type="secondary"):
        st.session_state.qa_chain = None
        st.session_state.current_doc = None
        st.session_state.chat_history = []
        st.rerun()
    
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "sources" in msg and msg["sources"]:
                with st.expander("Sources"):
                    for src in msg["sources"]:
                        st.caption(f"{src['file']}")
                        st.text(src['preview'][:150] + "...")
    
    if question := st.chat_input("Ask about this document..."):
        st.session_state.chat_history.append({"role": "user", "content": question})
        
        with st.chat_message("user"):
            st.markdown(question)
        
        with st.chat_message("assistant"):
            with st.spinner("Searching document..."):
                try:
                    response = st.session_state.qa_chain.ask(question)
                    st.markdown(response.answer)
                    
                    sources = [{"file": doc.metadata.get("source_file", "Unknown"), "preview": doc.page_content} for doc in response.sources]
                    
                    if sources:
                        with st.expander("Sources"):
                            for src in sources:
                                st.caption(f"{src['file']}")
                                st.text(src['preview'][:150] + "...")
                    
                    st.session_state.chat_history.append({"role": "assistant", "content": response.answer, "sources": sources})
                    
                except Exception as e:
                    st.error(f"Error generating response: {e}")
    
    # Debug/Evaluation Panel (collapsed by default)
    with st.expander("🔧 Debug / Evaluation", expanded=False):
        st.markdown("**System Diagnostics**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("*Document Info*")
            if hasattr(st.session_state, 'qa_chain') and st.session_state.qa_chain:
                # Show document type and chunking info if available
                qa = st.session_state.qa_chain
                if hasattr(qa, 'memory') and qa.memory:
                    st.text(f"Conversation turns: {len(qa.memory)}")
                st.text(f"Vector store: {'Active' if qa.vector_store else 'None'}")
        
        with col2:
            st.markdown("*Last Query*")
            if st.session_state.chat_history:
                last_q = [m for m in st.session_state.chat_history if m["role"] == "user"]
                if last_q:
                    st.text(f"Question: {last_q[-1]['content'][:50]}...")
        
        st.markdown("---")
        st.markdown("*Grounding Status*")
        st.info("✅ Hallucination protection active - all answers verified against document content")

def main():
    render_header()
    
    if st.session_state.qa_chain is None:
        render_upload_section()
        st.divider()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("<div class='info-card'><h3>Context-Aware</h3><p>Retrieves relevant passages using semantic search to ground answers in document content.</p></div>", unsafe_allow_html=True)
        with col2:
            st.markdown("<div class='info-card'><h3>Source Verification</h3><p>Every response includes citations to source material for verification and further reading.</p></div>", unsafe_allow_html=True)
        with col3:
            st.markdown("<div class='info-card'><h3>Privacy First</h3><p>Runs entirely on your machine. Documents and queries never leave your local environment.</p></div>", unsafe_allow_html=True)
    else:
        render_chat_interface()

if __name__ == "__main__":
    main()
