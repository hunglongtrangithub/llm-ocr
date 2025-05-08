import os
from pathlib import Path

import pymupdf
import pymupdf4llm

from src import config
from src.index.chunking import get_md_pages


def test_nccn():
    nccn_doc = pymupdf.open(config.RAW_DIR / "NCCNGuidelines.pdf")
    for page_id in range(nccn_doc.page_count):
        page = nccn_doc.load_page(page_id)
        textpage = page.get_textpage()
        text = textpage.extractText(sort=True)
        print(text)
        md_text = pymupdf4llm.to_markdown(nccn_doc, pages=[page_id])
        print(md_text)
    nccn_doc.close()


def test_unstructured_nccn():
    input_pdf_path = config.RAW_DIR / "NCCNGuidelines.pdf"
    from unstructured.chunking.basic import chunk_elements
    from unstructured.partition.pdf import partition_pdf

    print(f"Processing {input_pdf_path}...")
    elements = partition_pdf(str(input_pdf_path))
    print(f"Found {len(elements)} elements in the PDF.")

    chunks = chunk_elements(elements, max_characters=1500)
    for i, chunk in enumerate(chunks):
        with open("temp.txt") as f:
            string = f"Chunk {i}:\n\nCategory: {chunk.category}\n\nMetadata: {chunk.metadata.to_dict()}\n\n{chunk.text}"
            f.write(string)
        os.system("bat temp.txt")
        input(f"Press Enter to view page {i + 2}/{len(chunks)}...")
        os.system("clear")  # Clear screen before showing next page
    if Path("temp.txt").exists():
        os.remove("temp.txt")  # Remove the temporary file


def test_pymupdf_nccn():
    input_pdf_path = config.RAW_DIR / "NCCNGuidelines.pdf"
    output_dir_path = config.PROCESSED_DIR / "NCCNGuidelines"
    max_workers = 8

    if output_dir_path.exists():
        md_pages = [file.read_text() for file in output_dir_path.glob("*.txt")]
    else:
        md_pages = get_md_pages(input_pdf_path, output_dir_path, max_workers)
        for page_id, md_page in enumerate(md_pages):
            txt_path = output_dir_path / f"{page_id}.txt"
            txt_path.write_text(md_page)

    for i, md_text in enumerate(md_pages):
        with open("temp.md", "w") as f:
            f.write(md_text)
        os.system("bat temp.md")
        input(f"Press Enter to view page {i + 2}/{len(md_pages)}...")
        os.system("clear")  # Clear screen before showing next page
    if Path("temp.md").exists():
        os.remove("temp.md")  # Remove the temporary file

    return md_pages
