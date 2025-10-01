# #stagerag/config.py
from dataclasses import dataclass
from typing import List, Dict

@dataclass
class ConfidenceConfig:
    """Configuration for confidence evaluation"""
    
    # Component weights for final confidence score
    weights: Dict[str, float] = None
    
    # Answer quality thresholds
    min_answer_length_low: int = 10
    min_answer_length_medium: int = 50
    
    # Uncertainty ratio thresholds
    uncertainty_ratio_low: float = 0.05
    uncertainty_ratio_medium: float = 0.15
    uncertainty_ratio_high: float = 0.25
    
    # Negative phrases that indicate poor quality
    negative_phrases: List[str] = None
    
    # Uncertainty indicators
    uncertainty_indicators: List[str] = None
    
    # Stop words for relevance calculation
    stop_words: List[str] = None
    
    def __post_init__(self):
        if self.weights is None:
            self.weights = {
                'retrieval': 0.25,
                'basic_quality': 0.25,
                'relevance': 0.25,
                'uncertainty': 0.25
            }
        
        if self.negative_phrases is None:
            self.negative_phrases = [
                "i don't know",
                "i'm not sure",
                "i cannot answer",
                "insufficient information",
                "no information available",
                "unable to provide",
                "don't have enough",
                "cannot determine",
                "unclear",
                "uncertain"
            ]
        
        if self.uncertainty_indicators is None:
            self.uncertainty_indicators = [
                "might", "maybe", "perhaps", "possibly", "probably",
                "could be", "may be", "seems", "appears", "likely",
                "uncertain", "unclear", "unsure", "unknown",
                "potentially", "presumably", "allegedly", "supposedly"
            ]
        
        if self.stop_words is None:
            self.stop_words = {
                'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have',
                'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
                'may', 'might', 'can', 'this', 'that', 'these', 'those', 'i', 'you',
                'he', 'she', 'it', 'we', 'they', 'my', 'your', 'his', 'her', 'its',
                'our', 'their', 'me', 'him', 'her', 'us', 'them'
            }

@dataclass 
class EvaluationConfig:
    """Configuration for evaluation settings"""
    
    # Dataset settings
    ms_marco_samples: int = 250
    natural_questions_samples: int = 250
    max_total_samples: int = 500
    
    # Evaluation metrics
    use_exact_match: bool = True
    use_f1_score: bool = True  
    use_bleu_score: bool = True
    use_bert_score: bool = True
    
    # Statistical testing
    significance_level: float = 0.05
    min_samples_for_test: int = 5
    
    # Output settings
    save_detailed_results: bool = True
    save_summary_plots: bool = True
    output_format: str = "csv"  # csv, json, both
    
    # Performance tracking
    track_memory_usage: bool = True
    track_response_time: bool = True
    track_confidence_scores: bool = True

@dataclass
class ModelConfig:
    """Configuration for model settings"""
    
    # Model names
    stagerag_1b_model: str = "meta-llama/Llama-3.2-1B-Instruct"
    stagerag_3b_model: str = "meta-llama/Llama-3.2-3B-Instruct"
    baseline_8b_model: str = "meta-llama/Llama-3.2-8B-Instruct"
    
    # Generation parameters
    temperature: float = 0.7
    top_p: float = 0.85
    max_new_tokens: int = 512
    max_seq_len: int = 2048
    repetition_penalty: float = 1.1
    
    # Optimization settings
    use_4bit_quantization: bool = False
    use_8bit_quantization: bool = False
    use_gradient_checkpointing: bool = False
    
    # Cache settings
    enable_cache: bool = True
    cache_size: int = 1000

@dataclass
class RAGConfig:
    """Configuration for RAG settings"""
    
    # Embedding model
    embedding_model: str = "all-MiniLM-L6-v2"
    
    # Retrieval parameters
    top_k: int = 3
    similarity_threshold: float = 0.3
    max_context_length: int = 1000
    
    # Index settings
    index_type: str = "flat"  # flat, ivf, hnsw
    normalize_embeddings: bool = True
    
    # Dataset settings
    knowledge_base_path: str = ""
    preprocess_documents: bool = True
    chunk_size: int = 512
    chunk_overlap: int = 50