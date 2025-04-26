

import marimo

__generated_with = "0.13.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import fitz  # PyMuPDF
    from pathlib import Path
    import polars as pl

    pat_docs_lens = []
    for _file in Path('./all_files_merged').glob('*.pdf'):
        # print(file)
        _pat_doc = fitz.open(_file)
        pat_docs_lens.append(len(_pat_doc))
        _pat_doc.close()
    pat_docs_lens_sr = pl.Series('lens', pat_docs_lens)
    print(pat_docs_lens_sr.describe())
    print(sum(pat_docs_lens))

    return (fitz,)


@app.cell
def _(fitz):
    nccn_file_name = './NCCNGuidelines.pdf'
    nccn_doc = fitz.open(nccn_file_name)
    nccn_texts = []
    for _p in nccn_doc.pages():
        nccn_texts.append(_p.get_text())
    nccn_doc.close()
    print(f"Number of pages: {len(nccn_texts)}")
    print(f"Number of words: {sum([len(_p.split()) for _p in nccn_texts])}")
    return (nccn_file_name,)


@app.cell
def _(fitz):
    pat_file_name = './all_files_merged/TCGA-2E-A9G8.921E6140-A03E-4FBD-9FB8-554AE96FD16C.pdf'
    test_pat_doc = fitz.open(pat_file_name)
    for _p in test_pat_doc.pages():
        print(_p.get_text())
    return


@app.cell
def _(nccn_file_name):
    import pymupdf4llm
    md_text = pymupdf4llm.to_markdown(nccn_file_name)
    print(md_text)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
