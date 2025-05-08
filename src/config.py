from pathlib import Path

ROOT_DIR = Path(__file__).parents[1]
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
REPORT_DIR = ROOT_DIR / "reports"
PROCESSED_DIR = DATA_DIR / "processed"
EMBEDDING_MODEL_NAME = "NeuML/pubmedbert-base-embeddings"