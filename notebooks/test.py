

import marimo

__generated_with = "0.13.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import pymupdf  # PyMuPDF
    from pathlib import Path
    import polars as pl
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
    nccn_file_name = config.RAW_DIR / "NCCNGuidelines.pdf"
    nccn_doc = pymupdf.open(nccn_file_name)
    nccn_texts = []
    for _p in nccn_doc.pages():
        nccn_texts.append(_p.get_text())
    nccn_doc.close()
    print(f"Number of pages: {len(nccn_texts)}")
    print(f"Number of words: {sum([len(_p.split()) for _p in nccn_texts])}")
    return (nccn_file_name,)


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
def _(nccn_file_name):
    import pymupdf4llm

    def check_pymupdf4llm():
        nccn_doc = pymupdf4llm.to_markdown(doc=nccn_file_name, pages=[0, 1], show_progress=True)
        return nccn_doc
    check_pymupdf4llm()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
