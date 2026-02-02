"""
Unit Tests - Bonus Feature

Comprehensive test suite for the RAG Document Q&A System.
Tests all major components and functionality.
"""

import unittest
import sys
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
import tempfile
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.loader import DocumentLoader, validate_document_path, compare_chunking_strategies
from src.vector_store import VectorStoreManager
from src.qa_chain import RAGChain, check_ollama_status, QAResponse
from src.conversation_memory import ConversationMemory, ConversationTurn, EnhancedRAGChainWithMemory
from src.evaluation import EvalMetrics, RAGEvaluator


class TestDocumentLoader(unittest.TestCase):
    """Test document loading and chunking."""
    
    def setUp(self):
        self.loader = DocumentLoader(chunk_size=100, chunk_overlap=20)
    
    def test_initialization(self):
        """Test loader initializes with correct parameters."""
        self.assertEqual(self.loader.chunk_size, 100)
        self.assertEqual(self.loader.chunk_overlap, 20)
        self.assertEqual(self.loader.chunking_strategy, "recursive")
    
    def test_supported_formats(self):
        """Test supported formats are defined."""
        formats = DocumentLoader.SUPPORTED_FORMATS
        self.assertIn('.pdf', formats)
        self.assertIn('.txt', formats)
        self.assertIn('.docx', formats)
        self.assertIn('.csv', formats)
    
    def test_validate_document_path_nonexistent(self):
        """Test validation fails for non-existent file."""
        result = validate_document_path("/nonexistent/file.pdf")
        self.assertFalse(result)
    
    def test_validate_document_path_unsupported(self):
        """Test validation fails for unsupported format."""
        with tempfile.NamedTemporaryFile(suffix='.xyz', delete=False) as tmp:
            result = validate_document_path(tmp.name)
            self.assertFalse(result)
        Path(tmp.name).unlink()
    
    def test_chunk_documents_empty(self):
        """Test chunking empty list returns empty."""
        result = self.loader.chunk_documents([])
        self.assertEqual(result, [])


class TestConversationMemory(unittest.TestCase):
    """Test conversation memory functionality."""
    
    def setUp(self):
        self.memory = ConversationMemory(max_history=5, session_id="test_123")
    
    def test_initialization(self):
        """Test memory initializes correctly."""
        self.assertEqual(self.memory.max_history, 5)
        self.assertEqual(self.memory.session_id, "test_123")
        self.assertEqual(len(self.memory.history), 0)
    
    def test_add_turn(self):
        """Test adding conversation turn."""
        self.memory.add_turn("What is RAG?", "RAG is Retrieval-Augmented Generation.")
        self.assertEqual(len(self.memory.history), 1)
        self.assertEqual(self.memory.history[0].question, "What is RAG?")
    
    def test_max_history_limit(self):
        """Test history respects max limit."""
        for i in range(7):
            self.memory.add_turn(f"Question {i}", f"Answer {i}")
        
        self.assertEqual(len(self.memory.history), 5)  # Should trim to max
        self.assertEqual(self.memory.history[-1].question, "Question 6")
    
    def test_get_context(self):
        """Test retrieving conversation context."""
        self.memory.add_turn("Q1", "A1")
        self.memory.add_turn("Q2", "A2")
        
        context = self.memory.get_context(window=2)
        self.assertIn("Q1", context)
        self.assertIn("A1", context)
    
    def test_get_stats(self):
        """Test conversation statistics."""
        self.memory.add_turn("What is ML?", "Machine Learning is...")
        self.memory.add_turn("What is AI?", "Artificial Intelligence is...", sources=[{"file": "test.pdf"}])
        
        stats = self.memory.get_stats()
        self.assertEqual(stats["total_turns"], 2)
        self.assertEqual(stats["sessions_with_sources"], 1)
        self.assertEqual(stats["source_citation_rate"], 50.0)
    
    def test_export_to_json(self):
        """Test JSON export functionality."""
        self.memory.add_turn("Test Q", "Test A")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test_export.json"
            result = self.memory.export_to_json(str(filepath))
            
            self.assertTrue(Path(result).exists())
            
            data = json.loads(filepath.read_text())
            self.assertEqual(data["metadata"]["session_id"], "test_123")
            self.assertEqual(len(data["conversation"]), 1)
    
    def test_clear(self):
        """Test clearing history."""
        self.memory.add_turn("Q", "A")
        self.memory.clear()
        self.assertEqual(len(self.memory.history), 0)


class TestEvalMetrics(unittest.TestCase):
    """Test evaluation metrics functionality."""
    
    def setUp(self):
        self.evaluator = RAGEvaluator()
    
    def test_metrics_initialization(self):
        """Test metrics dataclass."""
        metrics = EvalMetrics(
            relevance_score=0.85,
            coverage_score=0.75,
            latency_ms=120.5,
            chunk_count=3,
            avg_chunk_length=450
        )
        
        self.assertEqual(metrics.relevance_score, 0.85)
        self.assertEqual(metrics.to_dict()["relevance_score"], 0.85)
    
    def test_evaluate_retrieval_empty(self):
        """Test evaluation with empty chunks."""
        metrics = self.evaluator.evaluate_retrieval("test", [], None)
        self.assertEqual(metrics.relevance_score, 0.0)
    
    def test_evaluate_answer_quality(self):
        """Test answer quality evaluation."""
        answer = "RAG systems combine retrieval with generation."
        
        # Create mock source
        mock_source = Mock()
        mock_source.page_content = "Retrieval-Augmented Generation combines document retrieval with LLM generation."
        
        quality = self.evaluator.evaluate_answer_quality(answer, [mock_source])
        
        self.assertIn("grounding_ratio", quality)
        self.assertIn("answer_length", quality)
        self.assertGreater(quality["grounding_ratio"], 0.0)
    
    def test_evaluate_answer_quality_no_sources(self):
        """Test quality evaluation with no sources."""
        quality = self.evaluator.evaluate_answer_quality("Some answer", [])
        self.assertEqual(quality["grounding_ratio"], 0.0)
    
    def test_generate_report(self):
        """Test report generation."""
        metrics_list = [
            EvalMetrics(0.8, 0.7, 100, 3, 450),
            EvalMetrics(0.9, 0.8, 120, 3, 460)
        ]
        
        report = self.evaluator.generate_report(metrics_list)
        
        self.assertIn("RAG SYSTEM EVALUATION REPORT", report)
        self.assertIn("0.850", report)  # Average relevance


class TestQAChain(unittest.TestCase):
    """Test RAG QA Chain functionality."""
    
    @patch('src.qa_chain.requests.get')
    def test_check_ollama_status_running(self, mock_get):
        """Test Ollama status check when running."""
        mock_get.return_value = Mock(
            status_code=200,
            json=lambda: {"models": [{"name": "llama3.2:latest"}]}
        )
        
        result = check_ollama_status()
        self.assertEqual(result["status"], "running")
        self.assertIn("llama3.2", result["models"])
    
    @patch('src.qa_chain.requests.get')
    def test_check_ollama_status_not_running(self, mock_get):
        """Test Ollama status check when not running."""
        mock_get.side_effect = Exception("Connection refused")
        
        result = check_ollama_status()
        self.assertEqual(result["status"], "not_running")


class TestIntegration(unittest.TestCase):
    """Integration tests for full workflow."""
    
    def test_full_pipeline_mock(self):
        """Test full pipeline with mocked components."""
        # Create test document
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as tmp:
            tmp.write("RAG systems are important for document Q&A.")
            tmp_path = tmp.name
        
        try:
            # Load document
            loader = DocumentLoader(chunk_size=100, chunk_overlap=20)
            chunks = loader.load_and_chunk(tmp_path)
            
            self.assertGreater(len(chunks), 0)
            
            # Add to vector store (mock)
            with patch.object(VectorStoreManager, '__init__', lambda x, **kwargs: None):
                with patch.object(VectorStoreManager, 'add_documents', return_value=None):
                    vs = VectorStoreManager()
                    # Would normally add chunks here
                    
        finally:
            Path(tmp_path).unlink()


class TestChunkingComparison(unittest.TestCase):
    """Test chunking strategy comparison."""
    
    def test_compare_strategies(self):
        """Test comparison function exists and works."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as tmp:
            tmp.write("This is a test document. " * 50)
            tmp_path = tmp.name
        
        try:
            results = compare_chunking_strategies(
                tmp_path,
                strategies=["recursive"],
                sizes=[50, 100],
                overlaps=[10]
            )
            
            self.assertIn("recursive_50_10", results)
            self.assertIn("recursive_100_10", results)
            
        finally:
            Path(tmp_path).unlink()


def run_tests():
    """Run all tests and return results."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestDocumentLoader))
    suite.addTests(loader.loadTestsFromTestCase(TestConversationMemory))
    suite.addTests(loader.loadTestsFromTestCase(TestEvalMetrics))
    suite.addTests(loader.loadTestsFromTestCase(TestQAChain))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestChunkingComparison))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
