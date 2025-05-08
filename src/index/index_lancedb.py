from typing import cast

import numpy as np
import polars as pl
import sentence_transformers
import torch

import lancedb
from lancedb.embeddings import get_registry
from lancedb.embeddings.base import TextEmbeddingFunction
from lancedb.embeddings.registry import register
from lancedb.pydantic import LanceModel, Vector
from src import config

from .chunking import get_chunks

DB_URI = config.ROOT_DIR / "lancedb"
TABLE_NAME = "docs"


@register("sentence-transformer")
class CustomSentenceTransformer(TextEmbeddingFunction):
    def __init__(
        self, name: str = config.EMBEDDING_MODEL_NAME, device: str | None = None
    ):
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.model = sentence_transformers.SentenceTransformer(
            model_name_or_path=name, device=device
        )
        self.device = device

        ndims = self.model.get_sentence_embedding_dimension()
        if ndims is None:
            raise AssertionError(f"Embedding size of model {name} not known")
        self._ndims = ndims

    def generate_embeddings(self, texts: list[str]) -> list[np.ndarray]:
        embeddings = self.model.encode(
            texts,
            device=self.device,
            convert_to_numpy=True,
        )  # [len(texts), embedding_size]
        return list(embeddings)

    def ndims(self) -> int:
        return self._ndims


model = cast(
    CustomSentenceTransformer, get_registry().get("sentence-transformers").create()
)


class Docs(LanceModel):
    text: str = model.SourceField()
    vector: Vector(model.ndims()) = model.VectorField()  # type: ignore


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

    db = lancedb.connect(DB_URI)
    print(f"Created DB at {DB_URI}")

    if TABLE_NAME in db.table_names():
        if args.overwrite:
            print("Table docs already exists. Deleting...")
            db.drop_table(TABLE_NAME)
        else:
            print(f"Table {TABLE_NAME} already exists. Use --overwrite to overwrite.")
            exit(0)

    table = db.create_table(TABLE_NAME, schema=Docs)
    print(f"Created table {TABLE_NAME}")

    print("Extracting chunks")
    chunks = get_chunks(config.RAW_DIR / "NCCNGuidelines.pdf")

    df = pl.DataFrame({"text": chunks})
    print("Adding chunks to the table...")
    table.add(data=df)

    table_df = pl.from_arrow(table.to_arrow())
    print(table_df)

    print("Test: Querying the table...")
    query = "NCCN Categories of Preference"
    actual = table.search(query).limit(2)
    print(actual.to_polars())

    return table


if __name__ == "__main__":
    main()
