import argparse
import time
import random
import numpy as np
import torch
from stagerag import StageRAGSystem



def interactive_chat(stagerag_system: StageRAGSystem):
    """Interactive chat mode"""
    print("=== StageRAG Interactive Chat ===")
    print("Commands:")
    print("  'quit' or 'q' - Exit")
    print("  'mode speed' - Switch to speed mode")
    print("  'mode precision' - Switch to precision mode") 
    print("  'cache stats' - Show cache statistics")
    print("  'cache clear' - Clear cache")
    print("  'search <query>' - Test RAG retrieval")
    print("-" * 50)
    
    current_mode = "speed"
    
    while True:
        user_input = input(f"[{current_mode.upper()}] You: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            break
        elif user_input.lower() == 'mode speed':
            current_mode = "speed"
            print("Switched to Speed Mode (3-step pipeline)")
            continue
        elif user_input.lower() == 'mode precision':
            current_mode = "precision" 
            print("Switched to Precision Mode (4-step pipeline)")
            continue
        elif user_input.lower() == 'cache stats':
            stats = stagerag_system.cache.stats()
            print(f"Cache Stats: {stats}")
            continue
        elif user_input.lower() == 'cache clear':
            stagerag_system.cache.clear()
            print("Cache cleared successfully")
            continue
        elif user_input.lower().startswith('search '):
            query = user_input[7:].strip()
            if stagerag_system.rag_system:
                results = stagerag_system.rag_system.retrieve(query, top_k=3)
                print(f"\nSearch results for: {query}")
                if results:
                    for i, (conv_pair, score) in enumerate(results, 1):
                        print(f"Result {i} (Score: {score:.3f}):")
                        print(f"Q: {conv_pair.question}")
                        print(f"A: {conv_pair.answer[:200]}...")
                        print()
                else:
                    print("No relevant results found")
            else:
                print("RAG system not available")
            print("-" * 50)
            continue
        
        if not user_input:
            continue
        
        start_time = time.time()
        result = stagerag_system.process_query(user_input, mode=current_mode)
        end_time = time.time()
        
        print(f"\nAssistant: {result['answer']}")
        
        # Show system info
        confidence = result.get('confidence', {})
        print(f"\nSystem Info:")
        print(f"   Mode: {result.get('mode', 'unknown').upper()}")
        print(f"   Processing Time: {result.get('processing_time', end_time - start_time):.2f}s")
        print(f"   Confidence: {confidence.get('overall_confidence', 0.0):.3f} ({confidence.get('confidence_level', 'unknown')})")
        
        cache_stats = result.get('cache_stats', {})
        if cache_stats:
            print(f"   Cache Hit Rate: {cache_stats.get('hit_rate', '0%')}")
        
        intermediate = result.get('intermediate_results', {})
        if intermediate:
            print(f"   RAG Results: {intermediate.get('rag_count', 0)}")
        
        print("-" * 50)


def set_random(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    parser = argparse.ArgumentParser(description="StageRAG System Evaluation")
    
    # Model configuration
    parser.add_argument('--temperature', type=float, default=0.7, help='Sampling temperature')
    parser.add_argument('--top_p', type=float, default=0.85, help='Top-p sampling parameter')
    parser.add_argument('--max_new_tokens', type=int, default=512, help='Maximum new tokens to generate')
    parser.add_argument('--max_seq_len', type=int, default=2048, help='Maximum input sequence length')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use')
    
    # StageRAG specific configuration
    parser.add_argument('--use_4bit', action='store_true', help='Use 4-bit quantization')
    parser.add_argument('--cache_size', type=int, default=1000, help='Cache size for results')
    
    # RAG configuration
    parser.add_argument('--rag_dataset', type=str, required=True, help='Path to JSONL file for RAG knowledge base')
    parser.add_argument('--disable_rag', action='store_true', help='Disable RAG functionality')
    parser.add_argument('--rag_threshold', type=float, default=0.3, help='Minimum similarity score for RAG retrieval')
    
    # Evaluation mode
    parser.add_argument('--mode', type=str, choices=['interactive', 'batch'], default='interactive', help='Evaluation mode')
    parser.add_argument('--default_mode', type=str, choices=['speed', 'precision'], default='speed', help='Default processing mode')
    parser.add_argument('--compare_modes', action='store_true', help='Compare speed vs precision modes')
    parser.add_argument('--manual_input', action='store_true', help='Allow manual input in batch mode')
    
    # Other parameters
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seed
    set_random(args.seed)
    
    # Initialize StageRAG system
    print("Initializing StageRAG system...")
    stagerag_system = StageRAGSystem(args)
    
    print(f"\nStageRAG system ready!")
    print(f"Mode options: Speed (3-step) | Precision (4-step)")
    print(f"4-bit quantization: {'Enabled' if args.use_4bit else 'Disabled'}")
    print(f"RAG system: {'Enabled' if stagerag_system.rag_system else 'Disabled'}")
    print(f"Cache enabled: True (size: {args.cache_size})")
    
    # Run evaluation
    if args.mode == 'interactive':
        interactive_chat(stagerag_system)
    elif args.mode == 'batch':
        batch_evaluation(stagerag_system, args)

if __name__ == "__main__":
    main()