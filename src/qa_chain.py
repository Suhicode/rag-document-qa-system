"""
RAG pipeline: retrieve chunks, generate answer, cite sources.
"""

from typing import List, Dict, Any
from dataclasses import dataclass

from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate

from vector_store import VectorStoreManager


@dataclass
class QAResponse:
    """Answer with source citations."""
    answer: str
    sources: List[Document]
    query: str


class RAGChain:
    """
    RAG pipeline using Ollama (local LLM) and ChromaDB.
    
    Using Ollama because it runs locally without API keys.
    Custom prompt constrains LLM to only use provided context.
    """
    
    # Prompt template - {context} gets filled with top-k chunks, {question} is user query
    RAG_PROMPT_TEMPLATE = """You are a precise document assistant that answers questions STRICTLY based on the provided context.

CONTEXT:
---------------------
{context}
---------------------

QUESTION: {question}

CRITICAL INSTRUCTIONS:
1. Answer ONLY using information from the context above
2. If the context does not contain the answer, respond EXACTLY: "The document does not contain information about this."
3. Do NOT use any external knowledge or make assumptions
4. Be concise and factual
5. If you cite information, it must be directly from the context

Answer:"""

    def __init__(
        self,
        vector_store_manager: VectorStoreManager,
        model_name: str = "llama3.2",
        temperature: float = 0.0
    ):
        """Initialize RAG chain. Temperature 0.0 for deterministic answers."""
        self.vector_store = vector_store_manager
        self.model_name = model_name
        
        self.llm = Ollama(
            model=model_name,
            temperature=temperature,
            system="""You are a precise document assistant. 
Use ONLY the provided context to answer.
Never make up information not in the context."""
        )
        
        self.prompt = PromptTemplate(
            template=self.RAG_PROMPT_TEMPLATE,
            input_variables=["context", "question"]
        )
        
        self.qa_chain = self._build_chain()
    
    def _build_chain(self) -> RetrievalQA:
        """Build RetrievalQA chain with top-3 retrieval."""
        retriever = self.vector_store.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": self.prompt}
        )
        
        return qa_chain
    
    def ask(self, question: str) -> QAResponse:
        """Query the RAG system and return answer with sources."""
        if not question or not question.strip():
            raise ValueError("Question cannot be empty")
        
        if self.vector_store.vector_store is None:
            raise ValueError("No documents loaded. Please load documents first.")
        
        try:
            result = self.qa_chain.invoke({"query": question})
            answer = result["result"]
            sources = result["source_documents"]
            
            # Verify answer is grounded in sources
            if not self._is_answer_grounded(answer, sources, question):
                search_context = self._generate_refusal_context(question, sources)
                answer = f"I couldn't find specific information about this in the document. {search_context}"
            
            return QAResponse(
                answer=answer,
                sources=sources,
                query=question
            )
            
        except Exception as e:
            error_msg = str(e).lower()
            if "connection" in error_msg:
                raise RuntimeError(
                    "Cannot connect to Ollama. Please ensure:"
                    "\n1. Ollama is installed (ollama.com/download)"
                    "\n2. Ollama server is running"
                    "\n3. Model is pulled: ollama pull llama3.2"
                ) from e
            raise
    
    def _is_answer_grounded(self, answer: str, sources: List[Document], question: str = "") -> bool:
        """
        Check if answer is grounded in sources.
        
        Uses entity matching (for resumes) and word overlap (general docs).
        Returns True if 30%+ of meaningful words appear in sources.
        """
        if "does not contain information" in answer.lower():
            return True
        
        if not sources:
            return False
        
        source_text = " ".join([doc.page_content.lower() for doc in sources])
        answer_lower = answer.lower()
        
        # Try entity matching first (works better for resumes)
        import re
        entity_pattern = r'\b[A-Z][a-zA-Z]*\b|\b[a-z]+[+\.#]*\d*\b'
        answer_entities = set(re.findall(entity_pattern, answer))
        common_words = {"The", "A", "An", "This", "That", "These", "Those", "Is", "Are", "Was", "Were", "Be", "Been", "Have", "Has", "Had", "Do", "Does", "Did", "Will", "Would", "Could", "Should", "May", "Might", "Can"}
        answer_entities = answer_entities - common_words
        
        if answer_entities:
            entities_in_sources = sum(1 for entity in answer_entities if entity.lower() in source_text)
            entity_coverage = entities_in_sources / len(answer_entities) if answer_entities else 0
            if entity_coverage >= 0.5:
                return True
        
        # Fallback: word overlap
        answer_words = set(answer_lower.split())
        stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by", "is", "are", "was", "were", "be", "been", "have", "has", "had", "do", "does", "did", "will", "would", "could", "should", "may", "might", "can", "this", "that", "these", "those", "i", "you", "he", "she", "it", "we", "they"}
        meaningful_answer_words = answer_words - stop_words
        
        if not meaningful_answer_words:
            return True
        
        words_in_sources = sum(1 for word in meaningful_answer_words if word in source_text)
        coverage_ratio = words_in_sources / len(meaningful_answer_words)
        
        return coverage_ratio >= 0.3
    
    def _generate_refusal_context(self, question: str, sources: List[Document]) -> str:
        """Generate helpful message about what was searched when answer isn't found."""
        import re
        
        # Extract key entities from question
        question_lower = question.lower()
        
        # Check for skill/technology questions
        skill_keywords = ['skill', 'technology', 'programming', 'language', 'framework', 'tool', 'platform']
        if any(kw in question_lower for kw in skill_keywords):
            return "I searched for skills and technologies mentioned in the document, but didn't find a clear match."
        
        # Check for experience/work history questions
        exp_keywords = ['experience', 'work', 'job', 'position', 'company', 'employer', 'career', 'role']
        if any(kw in question_lower for kw in exp_keywords):
            return "I looked for work experience or employment details, but couldn't locate specific information."
        
        # Check for education questions
        edu_keywords = ['education', 'degree', 'university', 'college', 'school', 'qualification', 'certification']
        if any(kw in question_lower for kw in edu_keywords):
            return "I searched for educational background or qualifications, but didn't find relevant details."
        
        # Check for contact/info questions
        contact_keywords = ['contact', 'email', 'phone', 'address', 'location', 'name']
        if any(kw in question_lower for kw in contact_keywords):
            return "I looked for contact information or personal details, but couldn't find what you're asking about."
        
        # Check if any sources were retrieved
        if sources:
            return f"I found {len(sources)} potentially relevant section(s), but they don't contain a clear answer to your specific question."
        else:
            return "No relevant sections were found in the document."
    
    def format_response_with_sources(self, response: QAResponse, max_sources: int = 3) -> str:
        """Format answer with source citations. De-duplicates sources."""
        lines = [
            "=" * 60,
            "ANSWER",
            "=" * 60,
            response.answer,
            "",
            "=" * 60,
            "SOURCES",
            "=" * 60,
        ]
        
        # De-duplicate by content signature
        unique_sources = []
        seen_content = set()
        
        for doc in response.sources:
            content_sig = doc.page_content[:100].lower().strip()
            if content_sig not in seen_content:
                seen_content.add(content_sig)
                unique_sources.append(doc)
        
        sources_to_show = unique_sources[:max_sources]
        
        for i, doc in enumerate(sources_to_show, 1):
            source_file = doc.metadata.get("source_file", "Unknown")
            page = doc.metadata.get("page", "N/A")
            
            lines.append(f"\n[{i}] {source_file}")
            if page != "N/A":
                lines.append(f"    Page: {page}")
            
            content = doc.page_content.replace("\n", " ").strip()
            if len(content) > 150:
                content = content[:150] + "..."
            lines.append(f"    Content: {content}")
        
        if not sources_to_show:
            lines.append("\nNo sources retrieved - answer may not be grounded in document.")
        elif len(unique_sources) > max_sources:
            lines.append(f"\n... and {len(unique_sources) - max_sources} more similar sources")
        
        lines.append("=" * 60)
        
        return "\n".join(lines)


def check_ollama_status() -> Dict[str, Any]:
    """Check if Ollama is running and return available models."""
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            return {
                "status": "running",
                "models": [m["name"] for m in models]
            }
        return {"status": "error", "message": "Unexpected response"}
    except Exception as e:
        return {
            "status": "not_running",
            "message": str(e)
        }
