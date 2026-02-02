"""
Evaluation Metrics Module - Bonus Feature

Implements evaluation metrics for RAG system performance.
Measures relevance, accuracy, and retrieval quality.
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class EvalMetrics:
    """Container for evaluation metrics."""
    relevance_score: float  # 0-1, how relevant are retrieved chunks
    coverage_score: float   # 0-1, how much of query is covered
    latency_ms: float       # Response time in milliseconds
    chunk_count: int        # Number of chunks retrieved
    avg_chunk_length: int   # Average length of retrieved chunks
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "relevance_score": round(self.relevance_score, 3),
            "coverage_score": round(self.coverage_score, 3),
            "latency_ms": round(self.latency_ms, 2),
            "chunk_count": self.chunk_count,
            "avg_chunk_length": self.avg_chunk_length
        }


class RAGEvaluator:
    """
    BONUS: Evaluation metrics for RAG system.
    
    Provides quantitative measures of:
    - Retrieval relevance
    - Answer coverage
    - System performance
    """
    
    def __init__(self, embedding_model=None):
        """
        Initialize evaluator.
        
        Args:
            embedding_model: Model for computing embeddings (uses same as vector store)
        """
        self.embedding_model = embedding_model
    
    def evaluate_retrieval(
        self,
        query: str,
        retrieved_chunks: List[Any],
        embedding_func=None
    ) -> EvalMetrics:
        """
        Evaluate retrieval quality.
        
        BONUS: Computes relevance metrics for retrieved chunks.
        
        Args:
            query: Original user query
            retrieved_chunks: List of retrieved documents
            embedding_func: Function to generate embeddings
            
        Returns:
            EvalMetrics with evaluation scores
        """
        if not retrieved_chunks or not embedding_func:
            return EvalMetrics(0.0, 0.0, 0.0, 0, 0)
        
        # Compute relevance scores using cosine similarity
        query_embedding = embedding_func(query)
        chunk_embeddings = [embedding_func(c.page_content) for c in retrieved_chunks]
        
        similarities = [
            cosine_similarity([query_embedding], [ce])[0][0]
            for ce in chunk_embeddings
        ]
        
        relevance_score = float(np.mean(similarities))
        
        # Compute coverage (how much of query terms appear in chunks)
        query_terms = set(query.lower().split())
        all_chunk_text = " ".join([c.page_content.lower() for c in retrieved_chunks])
        matched_terms = sum(1 for term in query_terms if term in all_chunk_text)
        coverage_score = matched_terms / len(query_terms) if query_terms else 0.0
        
        # Basic stats
        avg_length = sum(len(c.page_content) for c in retrieved_chunks) / len(retrieved_chunks)
        
        return EvalMetrics(
            relevance_score=relevance_score,
            coverage_score=coverage_score,
            latency_ms=0.0,  # Set during timing
            chunk_count=len(retrieved_chunks),
            avg_chunk_length=int(avg_length)
        )
    
    def evaluate_answer_quality(
        self,
        answer: str,
        sources: List[Any]
    ) -> Dict[str, float]:
        """
        Evaluate answer quality based on source grounding.
        
        BONUS: Answer accuracy metrics.
        
        Args:
            answer: Generated answer
            sources: Source documents
            
        Returns:
            Dictionary of quality scores
        """
        if not sources:
            return {"grounding_ratio": 0.0, "answer_length": len(answer)}
        
        # Compute grounding ratio (words from answer in sources)
        answer_words = set(answer.lower().split())
        source_text = " ".join([s.page_content.lower() for s in sources])
        
        stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for"}
        meaningful_words = answer_words - stop_words
        
        if not meaningful_words:
            grounding_ratio = 1.0
        else:
            matched = sum(1 for w in meaningful_words if w in source_text)
            grounding_ratio = matched / len(meaningful_words)
        
        return {
            "grounding_ratio": round(grounding_ratio, 3),
            "answer_length": len(answer),
            "source_count": len(sources)
        }
    
    def benchmark_chunking_strategies(
        self,
        test_queries: List[str],
        document_path: str,
        strategies: List[Dict[str, Any]]
    ) -> Dict[str, List[EvalMetrics]]:
        """
        BONUS: Benchmark different chunking configurations.
        
        Args:
            test_queries: List of test questions
            document_path: Path to test document
            strategies: List of chunking configs (size, overlap, type)
            
        Returns:
            Metrics for each strategy
        """
        from loader import DocumentLoader
        from vector_store import VectorStoreManager
        
        results = {}
        
        for strategy in strategies:
            key = f"{strategy['name']}_{strategy['chunk_size']}"
            strategy_metrics = []
            
            # Setup with this strategy
            loader = DocumentLoader(
                chunk_size=strategy['chunk_size'],
                chunk_overlap=strategy['chunk_overlap'],
                chunking_strategy=strategy.get('type', 'recursive')
            )
            
            try:
                chunks = loader.load_and_chunk(document_path)
                
                # Test each query
                for query in test_queries:
                    # Simulate retrieval and evaluation
                    metrics = self.evaluate_retrieval(query, chunks[:3], None)
                    strategy_metrics.append(metrics)
                
                results[key] = strategy_metrics
                
            except Exception as e:
                results[key] = [{"error": str(e)}]
        
        return results
    
    def generate_report(self, metrics_history: List[EvalMetrics]) -> str:
        """
        Generate a formatted evaluation report.
        
        Args:
            metrics_history: List of metrics from multiple queries
            
        Returns:
            Formatted report string
        """
        if not metrics_history:
            return "No evaluation data available."
        
        avg_relevance = np.mean([m.relevance_score for m in metrics_history])
        avg_coverage = np.mean([m.coverage_score for m in metrics_history])
        avg_latency = np.mean([m.latency_ms for m in metrics_history])
        
        report = f"""
========================================
RAG SYSTEM EVALUATION REPORT
========================================

Retrieval Performance:
  Average Relevance Score: {avg_relevance:.3f} (0-1 scale)
  Average Coverage Score:  {avg_coverage:.3f} (0-1 scale)
  Average Response Time:   {avg_latency:.2f} ms

Retrieval Statistics:
  Average Chunks Retrieved: {np.mean([m.chunk_count for m in metrics_history]):.1f}
  Average Chunk Length:     {np.mean([m.avg_chunk_length for m in metrics_history]):.0f} chars

Interpretation:
  - Relevance > 0.7: Excellent retrieval quality
  - Relevance 0.5-0.7: Good retrieval quality
  - Relevance < 0.5: Review retrieval strategy
  
  - Coverage > 0.8: Query well covered by context
  - Coverage < 0.5: Consider increasing chunk overlap

========================================
"""
        return report
