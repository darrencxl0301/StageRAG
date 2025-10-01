# StageRAG: Multi-Stage RAG System with Confidence Evaluation

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![HuggingFace](https://img.shields.io/badge/🤗-Dataset-yellow)](https://huggingface.co/datasets/darren0301/domain-mix-qa-1k)

A production-ready Retrieval-Augmented Generation (RAG) system using staged processing with Llama 3.2 models (1B and 3B). Features dual-mode processing (speed vs precision), LRU caching, and confidence-based response filtering.

## 🌟 Features

- **Dual Processing Modes**
  - **Speed Mode**: 3-step pipeline (1B + 3B models, ~3-4s response)
  - **Precision Mode**: 4-step pipeline (3B model, ~4-12s response)
- **Intelligent Caching**: LRU cache with configurable size for repeated queries
- **Confidence Evaluation**: Multi-component scoring system
- **Dynamic RAG Retrieval**: Adjusts result count based on confidence
- **4-bit Quantization**: Reduce memory usage while maintaining quality

## 📋 Prerequisites

### 1. Get Llama Model Access

You **must** request access to both Llama models:

1. Visit https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct
2. Visit https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct
3. Click "Access gated model" and accept the license
4. Wait for approval (usually instant)

### 2. Login to HuggingFace

```bash
pip install huggingface-hub
huggingface-cli login
# Enter your HuggingFace token when prompted
```

Get your token from: https://huggingface.co/settings/tokens

### 3. System Requirements

- Python >= 3.8
- CUDA-capable GPU (recommended) or CPU
- 6GB+ RAM for 4-bit mode, 8GB+ for full precision
- Internet connection for initial model download

## 🚀 Installation

### Clone Repository

```bash
git clone https://github.com/darrencxl0301/StageRAG.git
cd StageRAG
```

### Install Dependencies

```bash
# Install main dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

## 📊 Download Sample Dataset

### Option 1: Automatic Download (Recommended)

```bash
python scripts/download_data.py
```

This downloads the sample dataset from [darren0301/domain-mix-qa-1k](https://huggingface.co/datasets/darren0301/domain-mix-qa-1k) to `data/data.jsonl`.

### Option 2: Manual Download

```python
from datasets import load_dataset
import json

dataset = load_dataset("darren0301/domain-mix-qa-1k")

with open("data/data.jsonl", "w") as f:
    for item in dataset["train"]:
        json.dump({"conversations": item["conversations"]}, f)
        f.write("\n")
```

### Option 3: Use Your Own Data

Create a JSONL file with this format:

```json
{"conversations": [{"role": "user", "content": "What is EPF?"}, {"role": "assistant", "content": "EPF is the Employees Provident Fund..."}]}
{"conversations": [{"role": "user", "content": "How to apply for leave?"}, {"role": "assistant", "content": "To apply for leave..."}]}
```

## 💻 Usage

### Interactive Chat Demo

```bash
# Basic usage (CPU)
python demo/interactive_demo.py --rag_dataset data/data.jsonl

# With GPU and 4-bit quantization (recommended)
python demo/interactive_demo.py --rag_dataset data/data.jsonl --use_4bit --device cuda
```

**Interactive Commands:**
- `mode speed` - Switch to speed mode (3-step)
- `mode precision` - Switch to precision mode (4-step)
- `cache stats` - View cache performance
- `search <query>` - Test RAG retrieval
- `quit` or `q` - Exit

### Basic Usage Example

```bash
python demo/basic_usage.py --rag_dataset data/data.jsonl
```

### Programmatic Usage

```python
from stagerag import StageRAGSystem
import argparse

# Setup configuration
args = argparse.Namespace(
    rag_dataset='data/data.jsonl',
    device='cuda',
    use_4bit=True,
    cache_size=1000,
    temperature=0.7,
    top_p=0.85,
    max_new_tokens=512,
    max_seq_len=2048,
    disable_rag=False,
    rag_threshold=0.3,
    seed=42
)

# Initialize system
system = StageRAGSystem(args)

# Process query
result = system.process_query(
    "What are the EPF contribution rates?",
    mode="speed"
)

print(f"Answer: {result['answer']}")
print(f"Confidence: {result['confidence']['overall_confidence']:.3f}")
print(f"Time: {result['processing_time']:.2f}s")
```

## 🧪 Testing

```bash
# Install test dependencies
pip install pytest pytest-cov

# Run all tests
pytest tests/ -v

# Run specific test files
pytest tests/test_cache.py -v
pytest tests/test_confidence.py -v
pytest tests/test_rag.py -v

# Run with detailed output
pytest tests/test_cache.py -vv

# Run with coverage report
pytest tests/ --cov=stagerag --cov-report=html
```

## ⚙️ Configuration

### Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--rag_dataset` | Required | Path to JSONL knowledge base |
| `--device` | `cuda` | Device to use (cuda/cpu) |
| `--use_4bit` | `False` | Enable 4-bit quantization |
| `--cache_size` | `1000` | LRU cache size |
| `--temperature` | `0.7` | Sampling temperature (0.0-1.0) |
| `--top_p` | `0.85` | Top-p nucleus sampling |
| `--max_new_tokens` | `512` | Max tokens to generate |
| `--disable_rag` | `False` | Disable RAG retrieval |

### Confidence Weights

Edit `stagerag/config.py` to adjust confidence evaluation:

```python
weights = {
    'retrieval': 0.25,      # RAG retrieval quality
    'basic_quality': 0.25,  # Answer structure/length
    'relevance': 0.25,      # Keyword relevance
    'uncertainty': 0.25     # Uncertainty detection
}
```

## 📁 Project Structure

```
StageRAG/
├── stagerag/              # Main package
│   ├── __init__.py       # Package exports
│   ├── main.py           # StageRAGSystem class
│   ├── cache.py          # LRU cache implementation
│   ├── confidence.py     # Confidence evaluator
│   ├── rag.py            # RAG retrieval system
│   ├── prompts.py        # Prompt templates
│   └── config.py         # Configuration dataclasses
├── demo/                 # Usage examples
│   ├── interactive_demo.py
│   └── basic_usage.py
├── scripts/              # Utility scripts
│   └── download_data.py  # HuggingFace dataset downloader
├── tests/                # Test suite
│   ├── test_cache.py
│   ├── test_confidence.py
│   └── test_rag.py
├── data/                 # Knowledge base (created on first run)
│   └── data.jsonl
├── requirements.txt      # Production dependencies
├── requirements-dev.txt  # Development dependencies
├── setup.py             # Package configuration
└── README.md
```

## 🎯 Architecture

### Speed Mode (3-step Pipeline)
```
User Input → [1B] Normalize → [3B] RAG Filter → [1B] Generate Answer → Response
```

### Precision Mode (4-step Pipeline)
```
User Input → [1B] Normalize → [3B] RAG Retrieve → [3B] Synthesize → [3B] Final Answer → Response
```

## 📊 Performance Benchmarks

| Mode | Avg Time | Avg Confidence | Use Case |
|------|----------|----------------|----------|
| Speed | 3.3s | 0.68  | Real-time chat |
| Precision | 7.8s | 0.81 | Complex queries, critical decisions |

*Tested on NVIDIA RTX 3090 GPU with 4-bit quantization*

## 📦 Dataset

Sample dataset: [darren0301/domain-mix-qa-1k](https://huggingface.co/datasets/darren0301/domain-mix-qa-1k)

Contains 1,000 domain-specific Q&A pairs covering:
- Logical & Mathematical Reasoning 
- Specialized Medical Domain Knowledge
- Open-Ended General Instruction Following
- Employee benefits information
- Practical, Real-World Q&A

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 Citation

```bibtex
@software{stagerag2024,
  author = {Darren Chai Xin Lun},
  title = {StageRAG: Multi-Stage RAG System with Confidence Evaluation},
  year = {2024},
  url = {https://github.com/darrencxl0301/StageRAG},
  note = {Dataset: https://huggingface.co/datasets/darren0301/domain-mix-qa-1k}
}
```

## 🙏 Acknowledgments

- Built with [Llama 3.2](https://ai.meta.com/llama/) models by Meta
- [FAISS](https://github.com/facebookresearch/faiss) for vector similarity search
- [Sentence Transformers](https://www.sbert.net/) for embeddings
- [HuggingFace](https://huggingface.co/) for model hosting

## 📧 Contact

**Darren Chai Xin Lun**
- GitHub: [@darrencxl0301](https://github.com/darrencxl0301)
- HuggingFace: [@darren0301](https://huggingface.co/darren0301)

---

⭐ If you find this project helpful, please give it a star!
