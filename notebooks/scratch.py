

import marimo

__generated_with = "0.13.2"
app = marimo.App(width="medium")


@app.cell
def _():
    from pathlib import Path

    import polars as pl
    import pymupdf  # PyMuPDF

    from src import config

    pat_docs_lens = []
    for _file in Path(config.RAW_DIR / "TCGA_Reports_pdf").glob('*.pdf'):
        # print(file)
        _pat_doc = pymupdf.open(_file)
        pat_docs_lens.append(len(_pat_doc))
        _pat_doc.close()
    pat_docs_lens_sr = pl.Series('lens', pat_docs_lens)
    print(pat_docs_lens_sr.describe())
    print(sum(pat_docs_lens))

    return config, pymupdf


@app.cell
def _(config, pymupdf):
    def check_nccn_file():
        nccn_file_name = config.RAW_DIR / "NCCNGuidelines.pdf"
        nccn_doc = pymupdf.open(nccn_file_name)
        nccn_texts = []
        for _p in nccn_doc.pages():
            nccn_texts.append(_p.get_text())
        nccn_doc.close()
        print(f"Number of pages: {len(nccn_texts)}")
        print(f"Number of words: {sum([len(_p.split()) for _p in nccn_texts])}")

    check_nccn_file()
    return


@app.cell
def _(config, pymupdf):
    def check_pdf():
        pat_file_name = config.RAW_DIR / "TCGA_Reports_pdf" / 'TCGA-2E-A9G8.921E6140-A03E-4FBD-9FB8-554AE96FD16C.pdf'
        test_pat_doc = pymupdf.open(pat_file_name)
        c = 0
        for page in test_pat_doc.pages():
            print(page.get_text())
            print("\f")
            c += 1
            if c == 1:
                break

    check_pdf()
    return


@app.cell
def _(config, pymupdf):
    import csv

    def check_dict_pdf():
        doc = pymupdf.open(config.RAW_DIR / "NCCNGuidelines.pdf")
        toc = doc.get_toc()

        output_csv = config.PROCESSED_DIR / "NCCNGuidelines_TOC.csv"

        with open(output_csv, mode="w", newline='', encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["Level", "Title", "PageNumber"])  # Write header

            for sec in toc:
                writer.writerow([sec[0], sec[1], sec[2]])

        print(f"TOC written to {output_csv}")

    # check_dict_pdf()
    return


@app.cell
def _(config, pymupdf):
    def check_nccn_text():
        nccn_file_name = config.RAW_DIR / "NCCNGuidelines.pdf"
        nccn_doc = pymupdf.open(nccn_file_name)

        for page_id in range(nccn_doc.page_count):
            if page_id + 1 != 14:
                continue
            page = nccn_doc.load_page(page_id)
            textpage = page.get_textpage()
            text = textpage.extractText(sort=True)
            print(text)
        nccn_doc.close()

    check_nccn_text()
    return


@app.cell
def _():
    import faiss
    from sentence_transformers import SentenceTransformer

    from src.scripts.pdf_to_md import test_pymupdf_nccn

    def check_indexing():
        device = "mps"
        doc = "Hello, world!" * 1
        # Load the model
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)
        model_max_seq_len = model.get_max_seq_length() or 0
        print(f"Model's max sequence length: {model_max_seq_len}")
        c = 0
        tokenized_doc = model.tokenize([doc])
        doc_len = len(tokenized_doc["input_ids"][0])
        print(f"Tokenized doc length: {doc_len}")
        if len(tokenized_doc["input_ids"][0]) == model_max_seq_len:
            print("Too long sequence")

    
        # # Create a FAISS index
        # index = faiss.IndexFlatL2(embeddings.shape[1])
        # index.add(embeddings)

        # # Save the index to a file
        # faiss.write_index(index, config.PROCESSED_DIR / "nccn_index.index")    
    check_indexing()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
