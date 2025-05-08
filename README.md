# LLM-OCR: Enhanced Pathology Report Analysis with RAG

A research project exploring the enhancement of Large Language Model (LLM) responses to pathology report queries using Retrieval-Augmented Generation (RAG) with NCCN guidelines.

## Overview

This project investigates how RAG (Retrieval-Augmented Generation) can improve LLM responses when answering questions about pathology reports. We use NCCN (National Comprehensive Cancer Network) documents as our knowledge base, partitioning them into semantic chunks using Unstructured.io and indexing them with LanceDB for efficient retrieval.

The system aims to provide more accurate and contextually relevant responses by grounding the LLM's knowledge with specific NCCN guidelines, potentially improving the quality of medical information extraction and interpretation.


## Installation

1. Clone the repository:
```bash
git clone https://github.com/hunglongtrangithub/llm-ocr.git
cd llm-ocr
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Unix/macOS
# or
.venv\Scripts\activate  # On Windows
```

3. Install dependencies using uv:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh  # Install uv if not already installed
uv sync
```

## Usage

1. Prepare your NCCN document `NCCNGuidelines.pdf` in the `data/raw` directory.

2. Run the document processing pipeline:
```bash
python -m src.index.lancedb
```

3. Run the RAG demo:
```bash
python run.py
```

## Project Structure

```
.
├── data/           # NCCN document and pathology reports
├── lancedb/        # Vector database storage
├── notebooks/      # Jupyter notebooks for analysis
├── src/           # Source code
└── run.py         # Main entry point
```

## Results

TODO