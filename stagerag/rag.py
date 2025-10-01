#stagerag/rag.py
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class ConversationPair:
    question: str
    answer: str
    metadata: dict = None

class ConversationRAG:
    """RAG system for conversation retrieval"""
    
    def __init__(self, embedding_model="all-MiniLM-L6-v2"):
        self.embedding_model = SentenceTransformer(embedding_model)
        self.conversations = []
        self.index = None
        self.embeddings = None
    
    def load_conversations_from_jsonl(self, jsonl_path: str) -> List[ConversationPair]:
        """Load conversations from JSONL file"""
        conversations = []
        try:
            with open(jsonl_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        data = json.loads(line.strip())
                        if 'conversations' in data:
                            conversation_list = data['conversations']
                            user_msg = None
                            assistant_msg = None
                            
                            for msg in conversation_list:
                                if msg.get('role') == 'user':
                                    user_msg = msg.get('content', '').strip()
                                elif msg.get('role') == 'assistant':
                                    assistant_msg = msg.get('content', '').strip()
                                    break
                            
                            if user_msg and assistant_msg:
                                conversations.append(ConversationPair(
                                    question=user_msg,
                                    answer=assistant_msg,
                                    metadata={'line_number': line_num}
                                ))
                    except:
                        continue
        except Exception as e:
            print(f"Error reading file: {e}")
            return []
        
        print(f"Loaded {len(conversations)} conversation pairs")
        return conversations
    
    def build_index(self, jsonl_path: str) -> bool:
        """Build FAISS index for retrieval"""
        print("Loading conversations...")
        self.conversations = self.load_conversations_from_jsonl(jsonl_path)
        
        if not self.conversations:
            print("No conversations loaded. Cannot build index.")
            return False
        
        print("Generating embeddings...")
        questions = [conv.question for conv in self.conversations]
        self.embeddings = self.embedding_model.encode(questions, convert_to_tensor=False)
        self.embeddings = np.array(self.embeddings).astype('float32')
        
        print("Building FAISS index...")
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        faiss.normalize_L2(self.embeddings)
        self.index.add(self.embeddings)
        
        print(f"Index built with {self.index.ntotal} vectors")
        return True
    
    def retrieve(self, query: str, top_k: int = 3, threshold: float = 0.3, high_confidence_threshold: float = 0.95) -> List[Tuple]:
        """Retrieve relevant conversation pairs with dynamic result count based on confidence"""
        if self.index is None or not self.conversations:
            return []
        
        query_embedding = self.embedding_model.encode([query], convert_to_tensor=False)
        query_embedding = np.array(query_embedding).astype('float32')
        faiss.normalize_L2(query_embedding)
        
        scores, indices = self.index.search(query_embedding, top_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if score >= threshold and idx < len(self.conversations):
                results.append((self.conversations[idx], float(score)))
        
        # If top result has very high confidence (>95%), return only that result
        if results and results[0][1] >= high_confidence_threshold:
            return results[:1]  # Return only the top result
        
        # Otherwise, return multiple results as usual
        return results