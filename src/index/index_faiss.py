from pathlib import Path

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from src import config

from .chunking import get_chunks

INDEX_FILE_PATH = config.ROOT_DIR / "faiss"


def embed_texts(
    texts: list[str], model: SentenceTransformer, batch_size: int = 32
) -> np.ndarray:
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        device=model.device.type,
        convert_to_numpy=True,
    )  # [len(texts), embedding_size]
    print(
        f"Embeddings shape (num_embeddings * embedding_size): {embeddings.shape[0]} * {embeddings.shape[1]}"
    )
    print(
        f"Embeddings device: {embeddings.device}. Embeddings dtype: {embeddings.dtype}"
    )
    return embeddings


def index_faiss(embeddings: np.ndarray, index_file_path: Path) -> faiss.IndexFlatL2:
    if len(embeddings.shape) != 2:
        raise ValueError(
            f"Invalid embeddings shape {embeddings.shape}. Expected 2: num_embeddings * embedding_size"
        )
    num_embeddings, embedding_size = embeddings.shape
    index = faiss.IndexFlatL2(embedding_size)
    index.add(embeddings)  # type: ignore
    print(f"{num_embeddings} vectors added to the index")
    faiss.write_index(index, str(index_file_path))
    print(f"Index saved to {index_file_path}")
    return index


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--overwrite",
        "-o",
        action="store_true",
        help="Overwrite existing index if it exists",
    )
    args = parser.parse_args()

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device: {device}")
    model = SentenceTransformer(config.EMBEDDING_MODEL_NAME, device=device)
    print(f"Model {config.EMBEDDING_MODEL_NAME} initialized on device: {device}")

    if INDEX_FILE_PATH.exists():
        if args.overwrite:
            print(f"Index file {INDEX_FILE_PATH} already exists. Overwriting...")
        else:
            print(
                f"Index file {INDEX_FILE_PATH} already exists. Use --overwrite to overwrite."
            )
            exit(0)

    pdf_file_path = config.RAW_DIR / "NCCNGuidelines.pdf"
    texts = get_chunks(pdf_file_path)
    embeddings = embed_texts(texts, model)
    index = index_faiss(embeddings, INDEX_FILE_PATH)
    index = faiss.read_index(str(INDEX_FILE_PATH))

    query_texts = ["ABBREVIATIONS", "NCCN Categories of Preference"]
    query_embeddings = embed_texts(query_texts, model)
    k = 4
    print("Doing search:")
    scores, indexes = index.search(query_embeddings, k)  # type: ignore
    print(f"Scores: {scores}. Indexes: {indexes}")


if __name__ == "__main__":
    main()
