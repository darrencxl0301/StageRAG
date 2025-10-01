from setuptools import setup, find_packages
import os

# Read README if it exists, otherwise use description
long_description = ""
if os.path.exists("README.md"):
    with open("README.md", "r", encoding="utf-8") as f:
        long_description = f.read()

setup(
    name="stagerag",
    version="0.1.0",
    author="Darren Chai Xin Lun",
    description="A staged RAG system with confidence evaluation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/stagerag",
    packages=find_packages(),
    install_requires=[
        "torch>=2.2.0",
        "transformers>=4.48.0",
        "sentence-transformers>=2.3.0",
        "faiss-cpu>=1.7.4",
        "numpy>=1.26.0",
        "bitsandbytes>=0.41.0",
    ],
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)