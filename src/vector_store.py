"""
Vector Store Module

Handles embedding generation and storage using ChromaDB.
Manages the vector database lifecycle and similarity search.
"""

import os
from pathlib import Path
from typing import List, Optional, Dict, Any

from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma


class VectorStoreManager:
    """
    Manages document embeddings and vector storage using ChromaDB.
    
    DESIGN DECISION: all-MiniLM-L6-v2
    - Lightweight (80MB) yet powerful sentence transformer
    - 384-dimensional embeddings strike balance between quality and efficiency
    - Trained on diverse sentence pairs, good for semantic similarity
    - Runs locally, no API costs or rate limits
    - MIT license - safe for commercial use
    
    DESIGN DECISION: ChromaDB
    - Lightweight, embeddable vector database (no external server needed)
    - Persistent storage for document embeddings
    - Built-in cosine similarity search
    - Good integration with LangChain ecosystem
    """
    
    def __init__(
        self,
        persist_directory: str = "./chroma_db",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        """
        Initialize the vector store manager.
        
        Args:
            persist_directory: Where to store ChromaDB files
            embedding_model: HuggingFace model for embeddings
        """
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        # Initialize embeddings model
        # model_kwargs for device selection (CPU/GPU)
        # encode_kwargs for batch processing
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': 'cpu'},  # Use 'cuda' if GPU available
            encode_kwargs={'normalize_embeddings': True}  # L2 normalization for cosine similarity
        )
        
        self.vector_store: Optional[Chroma] = None
        self._initialize_vector_store()
    
    def _initialize_vector_store(self) -> None:
        """
        Initialize or load existing ChromaDB instance.
        """
        try:
            # Try to load existing collection first
            self.vector_store = Chroma(
                persist_directory=str(self.persist_directory),
                embedding_function=self.embeddings
            )
        except Exception:
            # If no existing DB, will be created on first add
            self.vector_store = None
    
    def clear_collection(self) -> None:
        """
        Clear all documents from the vector store.
        
        This removes the persistent storage and creates a fresh collection.
        """
        if self.vector_store is not None:
            try:
                # Get the client and delete the collection
                client = self.vector_store._client
                collection_name = self.vector_store._collection.name
                client.delete_collection(name=collection_name)
                
                # Close the connection
                self.vector_store = None
            except Exception:
                pass
        
        # Remove the persist directory to ensure clean start
        import shutil
        import time
        
        if self.persist_directory.exists():
            # Retry deletion with delay to handle file locks
            for attempt in range(3):
                try:
                    shutil.rmtree(self.persist_directory)
                    break
                except PermissionError:
                    if attempt < 2:
                        time.sleep(1)
                    else:
                        # If we can't delete, at least create a new directory with timestamp
                        timestamp = int(time.time())
                        new_dir = self.persist_directory.parent / f"chroma_db_{timestamp}"
                        self.persist_directory = new_dir
            
            self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        self.vector_store = None
    
    def add_documents(self, documents: List[Document]) -> None:
        """
        Add documents to the vector store.
        
        Documents are:
        1. Embedded using the sentence transformer
        2. Stored in ChromaDB with metadata
        3. Indexed for fast similarity search
        
        Args:
            documents: List of chunked documents to embed and store
        """
        if not documents:
            print("Warning: No documents to add")
            return
        
        # Clear existing collection before adding new documents
        self.clear_collection()
        
        # Create new ChromaDB with documents
        self.vector_store = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=str(self.persist_directory)
        )
        
        print(f"Added {len(documents)} document chunks to vector store")
    
    def similarity_search(
        self,
        query: str,
        k: int = 3
    ) -> List[Document]:
        """
        Retrieve top-k most similar documents to the query.
        
        Uses cosine similarity on normalized embeddings.
        
        Args:
            query: User question to search for
            k: Number of top results to retrieve
            
        Returns:
            List of most relevant document chunks
            
        Raises:
            ValueError: If vector store is empty
        """
        if self.vector_store is None:
            raise ValueError(
                "Vector store is empty. Please add documents first."
            )
        
        # similarity_search performs:
        # 1. Embed query using same model as documents
        # 2. Find k nearest neighbors using cosine similarity
        # 3. Return documents with similarity scores
        results = self.vector_store.similarity_search(
            query=query,
            k=k
        )
        
        return results
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector store.
        
        Returns:
            Dictionary with collection information
        """
        if self.vector_store is None:
            return {"status": "empty", "document_count": 0}
        
        try:
            count = self.vector_store._collection.count()
            return {
                "status": "active",
                "document_count": count,
                "persist_directory": str(self.persist_directory),
                "embedding_model": "sentence-transformers/all-MiniLM-L6-v2"
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def clear_store(self) -> None:
        """
        Clear all documents from the vector store.
        Use with caution - deletes all embeddings.
        """
        if self.vector_store is not None:
            # Chroma doesn't have a direct clear method
            # We delete and recreate the directory
            import shutil
            if self.persist_directory.exists():
                shutil.rmtree(self.persist_directory)
            self.persist_directory.mkdir(parents=True, exist_ok=True)
            self.vector_store = None
            print("Vector store cleared")


def create_vector_store_from_documents(
    documents: List[Document],
    persist_directory: str = "./chroma_db"
) -> VectorStoreManager:
    """
    Convenience function to create and populate a vector store.
    
    Args:
        documents: List of documents to embed
        persist_directory: Storage location for ChromaDB
        
    Returns:
        Configured VectorStoreManager instance
    """
    manager = VectorStoreManager(persist_directory=persist_directory)
    manager.add_documents(documents)
    return manager
