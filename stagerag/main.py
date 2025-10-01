#stagerag/main.py
import os
import time
import warnings
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from typing import Dict, List

from .cache import LRUCache
from .confidence import ConfidenceEvaluator
from .rag import ConversationRAG
from .prompts import PromptTemplates
from .config import ConfidenceConfig

class StageRAGSystem:
    """Main StageRAG system with improved architecture"""
    
    def __init__(self, args):
        self.args = args
        self.cache = LRUCache(max_size=args.cache_size)
        self.confidence_evaluator = ConfidenceEvaluator(ConfidenceConfig())
        self.prompts = PromptTemplates()
        
        # Initialize models
        print("Initializing models...")
        self._init_models()
        
        # Initialize RAG system
        self.rag_system = None
        if not args.disable_rag and args.rag_dataset:
            self._init_rag_system()
        
        # Warmup models
        self._warmup_models()
    
    def _init_models(self):
        """Initialize 1B and 3B models with optional quantization"""
        quantization_config = None
        if self.args.use_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        
        # Load 1B model
        print("Loading 1B model...")
        self.tokenizer_1b = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
        self.model_1b = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-3.2-1B-Instruct",
            quantization_config=quantization_config,
            torch_dtype=torch.float16 if not self.args.use_4bit else None
        ).to(self.args.device)
        
        # Load 3B model
        print("Loading 3B model...")
        self.tokenizer_3b = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
        self.model_3b = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-3.2-3B-Instruct",
            quantization_config=quantization_config,
            torch_dtype=torch.float16 if not self.args.use_4bit else None
        ).to(self.args.device)
        
        # Set pad tokens
        if self.tokenizer_1b.pad_token is None:
            self.tokenizer_1b.pad_token = self.tokenizer_1b.eos_token
        if self.tokenizer_3b.pad_token is None:
            self.tokenizer_3b.pad_token = self.tokenizer_3b.eos_token
        
        # Set to eval mode
        self.model_1b.eval()
        self.model_3b.eval()
        
        print(f"1B Model Parameters: {sum(p.numel() for p in self.model_1b.parameters()) / 1e6:.2f}M")
        print(f"3B Model Parameters: {sum(p.numel() for p in self.model_3b.parameters()) / 1e6:.2f}M")
    
    def _init_rag_system(self):
        """Initialize RAG system"""
        try:
            print("Setting up RAG system...")
            if not os.path.exists(self.args.rag_dataset):
                print(f"RAG dataset not found: {self.args.rag_dataset}")
                return
            
            self.rag_system = ConversationRAG()
            if self.rag_system.build_index(self.args.rag_dataset):
                print("RAG system initialized successfully!")
            else:
                self.rag_system = None
                print("Failed to build RAG index...")
        except Exception as e:
            print(f"Error initializing RAG system: {e}")
            self.rag_system = None
    
    def _warmup_models(self):
        """Warmup models to reduce first inference latency"""
        print("Warming up models...")
        dummy_prompts = ["Hello world", "Test prompt", "Warmup sequence"]
        
        for prompt in dummy_prompts:
            try:
                _ = self._generate_with_model(self.model_1b, self.tokenizer_1b, prompt, max_tokens=5)
                _ = self._generate_with_model(self.model_3b, self.tokenizer_3b, prompt, max_tokens=5)
            except:
                pass
        print("Model warmup completed!")
    
    def _generate_with_model(self, model, tokenizer, prompt: str, max_tokens: int = 256, step_name: str = "generate") -> str:
        """Generate text with LRU caching support"""
        model_name = "1B" if model == self.model_1b else "3B"
        
        # Check cache first
        cached_result = self.cache.get(model_name, prompt, step_name)
        if cached_result is not None:
            return cached_result
        
        try:
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.args.max_seq_len
            ).to(self.args.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=self.args.temperature,
                    top_p=self.args.top_p,
                    do_sample=self.args.temperature > 0,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                    repetition_penalty=1.1
                )
            
            input_length = inputs['input_ids'].shape[1]
            generated_tokens = outputs[0][input_length:]
            result = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
            
            # Cache the result
            self.cache.set(model_name, prompt, step_name, result)
            
            return result
            
        except Exception as e:
            print(f"Error in generation: {e}")
            return f"[Generation Error: {e}]"
    
    def speed_mode_pipeline(self, user_input: str) -> Dict:
        """Speed mode: 3-step processing pipeline"""
        start_time = time.time()
        
        # Step 1: Input normalization (1B model)
        normalized_input = self._step1_normalize_input_1b(user_input)
        
        # Step 2: RAG retrieval + filtering (3B model)
        rag_results = []
        if self.rag_system:
            # Use dynamic retrieval with high confidence threshold
            rag_results = self.rag_system.retrieve(
                normalized_input, 
                top_k=3, 
                high_confidence_threshold=0.95
            )
        
        organized_knowledge = self._step2_filter_and_organize_3b(normalized_input, rag_results)
        
        # Step 3: Answer generation (1B model)
        raw_answer = self._step3_generate_answer_1b(normalized_input, organized_knowledge)
        
        # Confidence evaluation
        confidence_result = self.confidence_evaluator.evaluate_comprehensive(
            raw_answer, normalized_input, rag_results
        )
        
        # Apply response strategy
        final_response = self._apply_response_strategy(raw_answer, confidence_result)
        
        processing_time = time.time() - start_time
        
        return {
            'answer': final_response,
            'mode': 'speed',
            'processing_time': round(processing_time, 2),
            'confidence': confidence_result,
            'steps_completed': 3,
            'cache_stats': self.cache.stats(),
            'intermediate_results': {
                'normalized_input': normalized_input,
                'rag_count': len(rag_results),
                'raw_answer_length': len(raw_answer)
            }
        }
    
    def precision_mode_pipeline(self, user_input: str) -> Dict:
        """Precision mode: 4-step processing pipeline"""
        start_time = time.time()
        
        # Step 1: Detailed normalization (3B model)
        normalized_input = self._step1_normalize_input_1b(user_input)
        
        # Step 2: Enhanced RAG retrieval + structuring (3B model)
        rag_results = []
        if self.rag_system:
            # Use dynamic retrieval with high confidence threshold
            rag_results = self.rag_system.retrieve(
                normalized_input, 
                top_k=5, 
                high_confidence_threshold=0.95
            )
        
        # Step 3Combined: Synthesize answer and extract evidence (3B model)
        organized_knowledge = self._step3_synthesize_and_extract_3b(normalized_input, rag_results)
        
        # Step 4: Generate final answer (3B model)
        raw_answer = self._step4_generate_final_answer_3b(normalized_input, organized_knowledge)
        
        # Confidence evaluation
        confidence_result = self.confidence_evaluator.evaluate_comprehensive(
            raw_answer, normalized_input, rag_results
        )
        
        # Apply response strategy
        final_response = self._apply_response_strategy(raw_answer, confidence_result)
        
        processing_time = time.time() - start_time
        
        return {
            'answer': final_response,
            'mode': 'precision',
            'processing_time': round(processing_time, 2),
            'confidence': confidence_result,
            'steps_completed': 5,
            'cache_stats': self.cache.stats(),
            'intermediate_results': {
                'normalized_input': normalized_input,
                'rag_count': len(rag_results),
                'raw_answer_length': len(raw_answer)
            }
        }
    
    # Speed mode pipeline methods
    def _step1_normalize_input_1b(self, user_input: str) -> str:
        """Step 1: Basic input normalization using 1B model"""
        prompt = self.prompts.NORMALIZE_INPUT_1B.format(user_input=user_input)
        return self._generate_with_model(
            self.model_1b, self.tokenizer_1b, prompt, 
            max_tokens=128, step_name="normalize_1b"
        )
    
    def _step2_filter_and_organize_3b(self, question: str, rag_results: List) -> str:
        """Step 2: Filter and organize RAG results using 3B model"""
        if not rag_results:
            return "No relevant information found in knowledge base."
        
        # Format RAG results
        rag_text = "\n".join([
            f"Q: {item[0].question}\nA: {item[0].answer}\nScore: {item[1]:.3f}\n"
            for item in rag_results[:5]
        ])
        
        prompt = self.prompts.FILTER_ORGANIZE_3B.format(
            question=question, rag_text=rag_text
        )
        return self._generate_with_model(
            self.model_3b, self.tokenizer_3b, prompt,
            max_tokens=256, step_name="filter_organize_3b"
        )
    
    def _step3_generate_answer_1b(self, question: str, organized_knowledge: str) -> str:
        """Step 3: Generate answer using 1B model"""
        prompt = self.prompts.GENERATE_ANSWER_1B.format(
            question=question, organized_knowledge=organized_knowledge
        )
        return self._generate_with_model(
            self.model_1b, self.tokenizer_1b, prompt,
            max_tokens=256, step_name="generate_answer_1b"
        )
    
    # Precision mode pipeline methods

    def _step3_synthesize_and_extract_3b(self, question: str, retrieved_information: str) -> str:
        """Step 3: One-shot synthesize a draft answer and extract supporting evidence."""
        prompt = self.prompts.SYNTHESIZE_EXTRACT_3B.format(
            question=question,
            retrieved_information=retrieved_information
        )
        return self._generate_with_model(
            self.model_3b, self.tokenizer_3b, prompt,
            max_tokens=1536,  # Increased max_tokens for this larger combined task
            step_name="synthesize_extract_3b"
        )
    
    def _step4_generate_final_answer_3b(self, question: str, organized_knowledge: str) -> str:
        """Step 4: Generate final answer using 3B model"""
        prompt = self.prompts.FINAL_ANSWER_3B.format(
            question=question, organized_knowledge=organized_knowledge
        )
        return self._generate_with_model(
            self.model_3b, self.tokenizer_3b, prompt,
            max_tokens=512, step_name="final_answer_3b"
        )
    
    def _apply_response_strategy(self, raw_answer: str, confidence_result: Dict) -> str:
        """Apply response strategy based on confidence"""
        recommendation = confidence_result['recommendation']
        
        if recommendation == "provide_answer":
            return raw_answer
        elif recommendation == "provide_with_warning":
            return f"Based on available information: {raw_answer}\n\n(Note: Please verify this information from official sources for accuracy)"
        else:
            return "I apologize, but I cannot provide an accurate answer to your question with the available information. Please consider rephrasing your question or consulting relevant experts."
    
    def process_query(self, user_input: str, mode: str = "speed") -> Dict:
        """Main processing interface"""
        if not self._validate_input(user_input):
            return self._create_error_response("Invalid input")
        
        try:
            if mode == "speed":
                return self.speed_mode_pipeline(user_input)
            elif mode == "precision":
                return self.precision_mode_pipeline(user_input)
            else:
                return self._create_error_response("Invalid mode")
        except Exception as e:
            print(f"Processing error: {e}")
            return self._create_error_response("System processing error")
    
    def _validate_input(self, user_input: str) -> bool:
        """Validate user input"""
        if not user_input or not user_input.strip():
            return False
        if len(user_input) > 2000:
            return False
        return True
    
    def _create_error_response(self, error_message: str) -> Dict:
        """Create error response"""
        return {
            'answer': f"Sorry, {error_message}. Please reenter your question.",
            'mode': 'error',
            'confidence': {'overall_confidence': 0.0, 'confidence_level': 'very_low'},
            'error': True
        }