from sentence_transformers import SentenceTransformer

from src import config
from src.index.chunking import get_chunks


def test_embed_chunks():
    pdf_file_path = config.RAW_DIR / "NCCNGuidelines.pdf"
    device = "mps"

    # Adjust these params so there are no truncated tokenized texts
    max_characters = 500
    overlap = 0

    texts = get_chunks(
        pdf_file_path,
        max_characters=max_characters,
        overlap=overlap,
    )
    model = SentenceTransformer(config.EMBEDDING_MODEL_NAME, device=device)
    tokenized_texts: dict[str, list[list[int]]] = model.tokenizer(texts)
    token_lengths = [len(tokens) for tokens in tokenized_texts["input_ids"]]
    print(token_lengths)
    max_seq_len = model.max_seq_length
    truncated_count = sum(1 for length in token_lengths if length > max_seq_len)
    print(f"Truncated count: {truncated_count}/{len(texts)}")
    assert truncated_count == 0
