#demo/basic_usage.py
from stagerag import StageRAGSystem
import argparse
import torch

def main():
    # Setup args
    parser = argparse.ArgumentParser()
    parser.add_argument('--rag_dataset', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--top_p', type=float, default=0.85)
    parser.add_argument('--max_new_tokens', type=int, default=512)
    parser.add_argument('--max_seq_len', type=int, default=2048)
    parser.add_argument('--use_4bit', action='store_true')
    parser.add_argument('--cache_size', type=int, default=1000)
    parser.add_argument('--disable_rag', action='store_true')
    parser.add_argument('--rag_threshold', type=float, default=0.3)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    # Initialize system
    print("Initializing StageRAG system...")
    system = StageRAGSystem(args)
    
    # Process a query in speed mode
    print("\n=== Speed Mode ===")
    result = system.process_query(
        "A 70-year-old woman, gravida 5, para 5, comes to the physician for the evaluation of sensation of vaginal fullness for the last six months. During this period, she has had lower back and pelvic pain that is worse with prolonged standing or walking. The patient underwent a hysterectomy at the age of 35 years because of severe dysmenorrhea. She has type 2 diabetes mellitus and hypercholesterolemia. Medications include metformin and atorvastatin. Vital signs are within normal limits. Pelvic examination elicits a feeling of pressure on the perineum. Pelvic floor muscle and anal sphincter tone are decreased. Pelvic examination shows protrusion of posterior vaginal wall with Valsalva maneuver and vaginal discharge. Which of the following is the most likely diagnosis?",
        mode="speed"
    )
    
    print(f"Answer: {result['answer']}")
    print(f"Confidence: {result['confidence']['overall_confidence']:.3f}")
    print(f"Processing Time: {result['processing_time']:.2f}s")
    
    # Process same query in precision mode
    print("\n=== Precision Mode ===")
    result = system.process_query(
        "A 70-year-old woman, gravida 5, para 5, comes to the physician for the evaluation of sensation of vaginal fullness for the last six months. During this period, she has had lower back and pelvic pain that is worse with prolonged standing or walking. The patient underwent a hysterectomy at the age of 35 years because of severe dysmenorrhea. She has type 2 diabetes mellitus and hypercholesterolemia. Medications include metformin and atorvastatin. Vital signs are within normal limits. Pelvic examination elicits a feeling of pressure on the perineum. Pelvic floor muscle and anal sphincter tone are decreased. Pelvic examination shows protrusion of posterior vaginal wall with Valsalva maneuver and vaginal discharge. Which of the following is the most likely diagnosis?",
        mode="precision"
    )
    
    print(f"Answer: {result['answer']}")
    print(f"Confidence: {result['confidence']['overall_confidence']:.3f}")
    print(f"Processing Time: {result['processing_time']:.2f}s")

if __name__ == "__main__":
    main()