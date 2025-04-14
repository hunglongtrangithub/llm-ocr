# LLM OCR

A project that leverages Large Language Models (LLMs) to process and analyze OCR text extracted from scanned pathology reports using AWS Textract.

## Overview

This project combines AWS Textract's OCR capabilities with LLM prompting to enhance the extraction and interpretation of medical information from scanned pathology report PDFs.

Currently in Phase 1: AWS Textract Implementation

### Project Phases

1. **Phase 1 (Current)**: Text extraction from PDF documents using AWS Textract
2. **Phase 2 (Planned)**: LLM prompting for analysis of extracted text

## Prerequisites

- AWS Account with Textract access
- Python 3.x
- AWS CLI configured
- uv (Python package installer)

## Setup

1. Install uv:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Clone the repository:

```bash
git clone https://github.com/hunglongtrangithub/llm-ocr.git
cd llm-ocr
```

3. Create and activate virtual environment and install dependencies:

```bash
uv sync
source .venv/bin/activate  # On Unix/macOS
# or
.venv\Scripts\activate  # On Windows
```

5. Configure AWS credentials:

```bash
aws configure
```
