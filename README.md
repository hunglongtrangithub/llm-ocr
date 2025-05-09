# LLM-OCR: Enhanced Pathology Report Analysis with RAG

A research project exploring how Retrieval-Augmented Generation (RAG) can enhance Large Language Model (LLM) responses to pathology report queries by leveraging NCCN guidelines.

## Overview

This NLP research project investigates the impact of RAG on improving LLM responses to pathology report queries. By using NCCN (National Comprehensive Cancer Network) guidelines as a knowledge base, we aim to provide more accurate and contextually relevant medical information.

### Key Components

- **Document Processing**: Uses Unstructured.io for intelligent partitioning of NCCN guidelines into semantic chunks
- **Vector Storage**: Uses LanceDB for efficient storage and similarity search of document chunks
- **RAG Pipeline**: Retrieval-Augmented Generation implementation combining retrieved context with LLM generation
- **Evaluation Framework**: Uses DeepEval metrics to compare RAG-enhanced vs base LLM responses

## Results

Our evaluation on 10 pathology reports using DeepEval metrics shows:

| Metric | RAGChat | BaseChat |
|--------|---------|----------|
| Answer Relevancy | 0.99 | 0.98 |
| Contextual Relevancy | 0.47 | N/A |

## Installation

### Prerequisites
- Python 3.12+
- uv package manager
- NCCN Guidelines PDF (`NCCNGuidelines.pdf`)
- Pathology report PDFs

### Setup

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

3. Install dependencies:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh  # Install uv if not already installed
uv sync
```

## Usage

### Data Preparation

1. Place input files in their respective directories:
   - `data/raw/NCCNGuidelines.pdf`: NCCN Guidelines document
   - `data/raw/TCGA_Reports_pdf/`: Directory for pathology report PDFs

### Running the Pipeline

1. Process pathology reports:
```bash
python -m src.ocr.ocr_pymupdf
```

2. Index NCCN Guidelines in LanceDB:
```bash
python -m src.index.index_lancedb
```

3. Launch the Gradio demo:
```bash
python run.py
```

### Evaluation

Run the evaluation script to compare RAGChat vs BaseChat:
```bash
python evaluate.py
```

## Project Structure

```
.
├── data/               # Data directory
│   ├── raw/           # Input files
│   │   ├── NCCNGuidelines.pdf
│   │   └── TCGA_Reports_pdf/
│   └── processed/     # Processed outputs
│       └── TCGA_Reports_txt/
├── src/               # Source code
│   ├── chat/         # Chat implementations
│   ├── index/        # Indexing logic
│   └── ocr/          # OCR processing
├── lancedb/          # Vector database
├── notebooks/        # Analysis notebooks
├── reports/          # Evaluation results
├── evaluate.py       # Evaluation script
└── run.py           # Demo application
```

## Implementation Details

### RAG Pipeline
1. **Document Chunking**: NCCN guidelines are split into semantic chunks using Unstructured.io
2. **Vector Storage**: Chunks are embedded and stored in LanceDB for similarity search
3. **Query Processing**: 
   - Input pathology reports are used to retrieve relevant NCCN guideline chunks
   - Retrieved chunks are formatted with XML-like tags for context
   - LLM generates responses using both the report and retrieved context

### Evaluation Metrics
- **Answer Relevancy**: Measures how well responses address the query
- **Contextual Relevancy**: Assesses use of retrieved NCCN context
