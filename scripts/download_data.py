"""Download dataset from HuggingFace"""

import argparse
import json
import os
from datasets import load_dataset
from huggingface_hub import login


def download_and_convert(dataset_name, output_path, use_auth=False):
    """Download dataset from HuggingFace and convert to JSONL"""
    
    # Login if needed (for private datasets)
    if use_auth:
        print("Please login to HuggingFace:")
        login()
    
    print(f"Downloading dataset: {dataset_name}")
    try:
        # Load dataset
        dataset = load_dataset(dataset_name)
        
        # Create output directory if needed
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Convert to JSONL
        print(f"Converting to JSONL format...")
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in dataset['train']:
                # Adjust field names based on your dataset structure
                if 'conversations' in item:
                    json.dump({'conversations': item['conversations']}, f)
                elif 'question' in item and 'answer' in item:
                    # If dataset has question/answer format, convert it
                    conversations = [
                        {"role": "user", "content": item['question']},
                        {"role": "assistant", "content": item['answer']}
                    ]
                    json.dump({'conversations': conversations}, f)
                else:
                    # Try to use the item as-is
                    json.dump(item, f)
                f.write('\n')
        
        print(f"✓ Dataset saved to: {output_path}")
        print(f"  Total conversations: {len(dataset['train'])}")
        return True
        
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Download HuggingFace dataset")
    parser.add_argument(
        '--dataset',
        type=str,
        default='darren0301/domain-mix-qa-1k',
        help='HuggingFace dataset name'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/data.jsonl',
        help='Output JSONL file path'
    )
    parser.add_argument(
        '--auth',
        action='store_true',
        help='Login to HuggingFace (for private datasets)'
    )
    
    args = parser.parse_args()
    
    success = download_and_convert(args.dataset, args.output, args.auth)
    
    if success:
        print("\nYou can now run:")
        print(f"  python demo/interactive_demo.py --rag_dataset {args.output}")
    else:
        print("\nFailed to download dataset. Please check:")
        print("  1. Dataset name is correct")
        print("  2. You have internet connection")
        print("  3. Dataset is public or you're logged in (use --auth)")


if __name__ == "__main__":
    main()