#tests/test_rag.py
"""Tests for RAG retrieval system"""

import pytest
import tempfile
import json
import os
import numpy as np
from stagerag.rag import ConversationRAG, ConversationPair


class TestConversationRAG:
    """Test suite for ConversationRAG system"""
    
    @pytest.fixture
    def sample_jsonl_file(self):
        """Create a temporary JSONL file with sample conversations"""
        # Create temporary file
        temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.jsonl')
        
        # Sample conversations
        conversations = [
            {
                "conversations": [
                    {"role": "user", "content": "What is EPF?"},
                    {"role": "assistant", "content": "EPF stands for Employees Provident Fund, a retirement savings scheme in Malaysia."}
                ]
            },
            {
                "conversations": [
                    {"role": "user", "content": "What are EPF contribution rates?"},
                    {"role": "assistant", "content": "The EPF contribution rate is 11% for employees and 12-13% for employers depending on salary."}
                ]
            },
            {
                "conversations": [
                    {"role": "user", "content": "How to withdraw EPF?"},
                    {"role": "assistant", "content": "You can withdraw EPF upon retirement at age 55, or for specific purposes like housing or education."}
                ]
            },
            {
                "conversations": [
                    {"role": "user", "content": "What is SOCSO?"},
                    {"role": "assistant", "content": "SOCSO is the Social Security Organization that provides social security protection."}
                ]
            }
        ]
        
        # Write to file
        for conv in conversations:
            temp_file.write(json.dumps(conv) + '\n')
        
        temp_file.close()
        
        yield temp_file.name
        
        # Cleanup
        os.unlink(temp_file.name)
    
    @pytest.fixture
    def empty_jsonl_file(self):
        """Create an empty JSONL file"""
        temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.jsonl')
        temp_file.close()
        
        yield temp_file.name
        
        os.unlink(temp_file.name)
    
    @pytest.fixture
    def malformed_jsonl_file(self):
        """Create a JSONL file with malformed data"""
        temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.jsonl')
        
        # Write some malformed lines
        temp_file.write('{"invalid": "json"\n')  # Missing closing brace
        temp_file.write('not json at all\n')
        temp_file.write('{"conversations": []}\n')  # Empty conversations
        temp_file.write('{"conversations": [{"role": "user", "content": "Valid question"}]}\n')  # Missing assistant
        
        temp_file.close()
        
        yield temp_file.name
        
        os.unlink(temp_file.name)
    
    def test_initialization(self):
        """Test RAG system initialization"""
        rag = ConversationRAG(embedding_model="all-MiniLM-L6-v2")
        
        assert rag.embedding_model is not None
        assert rag.conversations == []
        assert rag.index is None
        assert rag.embeddings is None
    
    def test_load_conversations(self, sample_jsonl_file):
        """Test loading conversations from JSONL file"""
        rag = ConversationRAG()
        conversations = rag.load_conversations_from_jsonl(sample_jsonl_file)
        
        # Verify loaded conversations
        assert len(conversations) == 4
        assert isinstance(conversations[0], ConversationPair)
        assert conversations[0].question == "What is EPF?"
        assert "Employees Provident Fund" in conversations[0].answer
        assert conversations[0].metadata['line_number'] == 1
    
    def test_load_conversations_empty_file(self, empty_jsonl_file):
        """Test loading from empty file"""
        rag = ConversationRAG()
        conversations = rag.load_conversations_from_jsonl(empty_jsonl_file)
        
        assert len(conversations) == 0
    
    def test_load_conversations_malformed(self, malformed_jsonl_file):
        """Test loading with malformed data"""
        rag = ConversationRAG()
        conversations = rag.load_conversations_from_jsonl(malformed_jsonl_file)
        
        # Should skip malformed lines gracefully
        assert len(conversations) == 0  # No valid complete conversations
    
    def test_load_conversations_nonexistent_file(self):
        """Test loading from non-existent file"""
        rag = ConversationRAG()
        conversations = rag.load_conversations_from_jsonl("nonexistent_file.jsonl")
        
        assert len(conversations) == 0
    
    def test_build_index_success(self, sample_jsonl_file):
        """Test successful index building"""
        rag = ConversationRAG()
        success = rag.build_index(sample_jsonl_file)
        
        assert success is True
        assert rag.index is not None
        assert rag.embeddings is not None
        assert len(rag.conversations) == 4
        assert rag.index.ntotal == 4
        assert rag.embeddings.shape[0] == 4
    
    def test_build_index_empty_file(self, empty_jsonl_file):
        """Test index building with empty file"""
        rag = ConversationRAG()
        success = rag.build_index(empty_jsonl_file)
        
        assert success is False
        assert rag.index is None
    
    def test_build_index_nonexistent_file(self):
        """Test index building with non-existent file"""
        rag = ConversationRAG()
        success = rag.build_index("nonexistent.jsonl")
        
        assert success is False
    
    def test_retrieve_basic(self, sample_jsonl_file):
        """Test basic retrieval functionality"""
        rag = ConversationRAG()
        rag.build_index(sample_jsonl_file)
        
        # Search for EPF-related query
        results = rag.retrieve("What is EPF contribution?", top_k=3, threshold=0.3)
        
        assert len(results) > 0
        assert all(isinstance(item, tuple) for item in results)
        assert all(len(item) == 2 for item in results)
        assert all(isinstance(item[0], ConversationPair) for item in results)
        assert all(isinstance(item[1], float) for item in results)
        
        # Results should be sorted by score (descending)
        scores = [item[1] for item in results]
        assert scores == sorted(scores, reverse=True)
    
    def test_retrieve_high_confidence_single_result(self, sample_jsonl_file):
        """Test that high confidence returns only top result"""
        rag = ConversationRAG()
        rag.build_index(sample_jsonl_file)
        
        # Exact match query should have high confidence
        results = rag.retrieve(
            "What is EPF?",
            top_k=3,
            threshold=0.3,
            high_confidence_threshold=0.95
        )
        
        # If top result is very confident, should return only 1
        if results and results[0][1] >= 0.95:
            assert len(results) == 1
    
    def test_retrieve_with_threshold(self, sample_jsonl_file):
        """Test retrieval respects similarity threshold"""
        rag = ConversationRAG()
        rag.build_index(sample_jsonl_file)
        
        # High threshold should return fewer results
        results_low_threshold = rag.retrieve("EPF information", top_k=5, threshold=0.3)
        results_high_threshold = rag.retrieve("EPF information", top_k=5, threshold=0.8)
        
        assert len(results_high_threshold) <= len(results_low_threshold)
        
        # All results should meet threshold
        for _, score in results_high_threshold:
            assert score >= 0.8
    
    def test_retrieve_top_k_limit(self, sample_jsonl_file):
        """Test that top_k limits number of results"""
        rag = ConversationRAG()
        rag.build_index(sample_jsonl_file)
        
        results = rag.retrieve("EPF", top_k=2, threshold=0.1)
        
        assert len(results) <= 2
    
    def test_retrieve_no_index(self):
        """Test retrieval without building index first"""
        rag = ConversationRAG()
        results = rag.retrieve("test query", top_k=3)
        
        assert results == []
    
    def test_retrieve_unrelated_query(self, sample_jsonl_file):
        """Test retrieval with completely unrelated query"""
        rag = ConversationRAG()
        rag.build_index(sample_jsonl_file)
        
        results = rag.retrieve(
            "What is the weather like on Mars?",
            top_k=3,
            threshold=0.7
        )
        
        # Should return empty or very low scores
        assert len(results) == 0 or all(score < 0.7 for _, score in results)
    
    def test_retrieve_semantic_similarity(self, sample_jsonl_file):
        """Test semantic similarity matching"""
        rag = ConversationRAG()
        rag.build_index(sample_jsonl_file)
        
        # Different phrasing of same question
        results1 = rag.retrieve("What is EPF?", top_k=1, threshold=0.3)
        results2 = rag.retrieve("Tell me about EPF", top_k=1, threshold=0.3)
        results3 = rag.retrieve("Explain EPF", top_k=1, threshold=0.3)
        
        # All should retrieve similar/same documents
        assert len(results1) > 0
        assert len(results2) > 0
        assert len(results3) > 0
        
        # Top results should be about EPF
        assert "EPF" in results1[0][0].answer or "Provident Fund" in results1[0][0].answer
    
    def test_conversation_pair_structure(self, sample_jsonl_file):
        """Test ConversationPair data structure"""
        rag = ConversationRAG()
        conversations = rag.load_conversations_from_jsonl(sample_jsonl_file)
        
        conv = conversations[0]
        assert hasattr(conv, 'question')
        assert hasattr(conv, 'answer')
        assert hasattr(conv, 'metadata')
        assert isinstance(conv.question, str)
        assert isinstance(conv.answer, str)
        assert isinstance(conv.metadata, dict)
    
    def test_embedding_normalization(self, sample_jsonl_file):
        """Test that embeddings are L2 normalized"""
        rag = ConversationRAG()
        rag.build_index(sample_jsonl_file)
        
        # Check that embeddings are normalized (L2 norm should be ~1)
        norms = np.linalg.norm(rag.embeddings, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-5)
    
    def test_retrieve_score_range(self, sample_jsonl_file):
        """Test that similarity scores are in valid range"""
        rag = ConversationRAG()
        rag.build_index(sample_jsonl_file)
        
        results = rag.retrieve("EPF", top_k=5, threshold=0.0)
        
        for _, score in results:
            assert 0.0 <= score <= 1.0


class TestConversationPair:
    """Test ConversationPair dataclass"""
    
    def test_creation(self):
        """Test creating ConversationPair"""
        pair = ConversationPair(
            question="What is X?",
            answer="X is Y",
            metadata={"source": "test"}
        )
        
        assert pair.question == "What is X?"
        assert pair.answer == "X is Y"
        assert pair.metadata == {"source": "test"}
    
    def test_creation_without_metadata(self):
        """Test creating ConversationPair without metadata"""
        pair = ConversationPair(
            question="What is X?",
            answer="X is Y"
        )
        
        assert pair.question == "What is X?"
        assert pair.answer == "X is Y"
        assert pair.metadata is None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])