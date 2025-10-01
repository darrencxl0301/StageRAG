#stagerag/confidence.py
from typing import List, Dict
from .config import ConfidenceConfig

class ConfidenceEvaluator:
    """Configurable confidence evaluator"""
    
    def __init__(self, config: ConfidenceConfig = None):
        self.config = config or ConfidenceConfig()
    
    def evaluate_comprehensive(self, answer: str, question: str, rag_results: List) -> Dict:
        """Comprehensive confidence evaluation"""
        scores = {
            'retrieval': self._evaluate_retrieval_quality(rag_results),
            'basic_quality': self._evaluate_basic_quality(answer),
            'relevance': self._evaluate_keyword_relevance(answer, question),
            'uncertainty': self._evaluate_uncertainty_language(answer)
        }
        
        final_confidence = sum(
            scores[component] * self.config.weights[component] 
            for component in scores
        )
        
        return {
            'overall_confidence': round(final_confidence, 3),
            'confidence_level': self._get_confidence_level(final_confidence),
            'component_scores': scores,
            'recommendation': self._get_response_recommendation(final_confidence)
        }
    
    def _evaluate_retrieval_quality(self, rag_results: List) -> float:
        """Evaluate retrieval quality"""
        if not rag_results:
            return 0.1
        
        top1_score = rag_results[0][1] if len(rag_results) > 0 else 0.3
        result_count = len(rag_results)
        count_factor = min(result_count / 5.0, 1.0)
        
        return min(top1_score * 0.8 + count_factor * 0.2, 1.0)
    
    def _evaluate_basic_quality(self, answer: str) -> float:
        """Evaluate basic answer quality"""
        quality_score = 1.0
        
        # Length check using config
        if len(answer) < self.config.min_answer_length_low:
            quality_score *= 0.2
        elif len(answer) < self.config.min_answer_length_medium:
            quality_score *= 0.6
        
        # Check for negative indicators
        answer_lower = answer.lower()
        for phrase in self.config.negative_phrases:
            if phrase in answer_lower:
                quality_score *= 0.4
                break
        
        # Check repetition
        words = answer.split()
        if len(words) > 0:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio < 0.6:
                quality_score *= 0.7
        
        return quality_score
    
    def _evaluate_uncertainty_language(self, answer: str) -> float:
        """Detect uncertainty language using config"""
        answer_lower = answer.lower()
        uncertainty_count = sum(1 for indicator in self.config.uncertainty_indicators 
                              if indicator in answer_lower)
        
        words = answer.split()
        uncertainty_ratio = uncertainty_count / max(len(words), 1)
        
        if uncertainty_ratio > self.config.uncertainty_ratio_high:
            return 0.3
        elif uncertainty_ratio > self.config.uncertainty_ratio_medium:
            return 0.5
        elif uncertainty_ratio > self.config.uncertainty_ratio_low:
            return 0.7
        else:
            return 1.0
    
    def _evaluate_keyword_relevance(self, answer: str, question: str) -> float:
        """Evaluate keyword relevance"""
        def extract_keywords(text):
            words = set(word.lower() for word in text.split() 
                       if word.lower() not in self.config.stop_words)
            return words
        
        question_words = extract_keywords(question)
        answer_words = extract_keywords(answer)
        
        if not question_words:
            return 0.5
        
        overlap = len(question_words & answer_words)
        relevance_score = overlap / len(question_words)
        
        return 0.3 + relevance_score * 0.7
    
    def _get_confidence_level(self, score: float) -> str:
        """Get confidence level string"""
        if score >= 0.8:
            return "high"
        elif score >= 0.6:
            return "medium"
        elif score >= 0.4:
            return "low"
        else:
            return "very_low"
    
    def _get_response_recommendation(self, score: float) -> str:
        """Get response recommendation"""
        if score >= 0.6:
            return "provide_answer"
        elif score >= 0.4:
            return "provide_with_warning"
        else:
            return "apologize_and_refuse"