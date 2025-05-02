from src import config
import pymupdf
import pymupdf4llm
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import os


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


def test_unstructured_nccn(input_pdf_path: Path):
    from unstructured.partition.pdf import partition_pdf
    from unstructured.chunking.title import chunk_by_title
    from unstructured.chunking.basic import chunk_elements

    print(f"Processing {input_pdf_path}...")
    elements = partition_pdf(str(input_pdf_path))
    print(f"Found {len(elements)} elements in the PDF.")

    chunks = chunk_elements(elements, max_characters=1500)
    # chunks = chunk_by_title(elements, max_characters=1500)
    for i, chunk in enumerate(chunks):
        with open("temp.txt") as f:
            string = f"Chunk {i}:\n\nCategory: {chunk.category}\n\nMetadata: {chunk.metadata.to_dict()}\n\n{chunk.text}"
            f.write(string)
        os.system("bat temp.txt")
        input(f"Press Enter to view page {i + 2}/{len(chunks)}...")
        os.system("clear")  # Clear screen before showing next page
    if Path("temp.txt").exists():
        os.remove("temp.txt")  # Remove the temporary file


def process_page(args: tuple[Path, int]) -> tuple[int, str]:
    input_pdf_path, page_id = args
    nccn_doc = pymupdf.open(input_pdf_path)
    md_text = pymupdf4llm.to_markdown(nccn_doc, pages=[page_id])
    nccn_doc.close()
    return page_id, md_text


def test_pymupdf_nccn(
    input_pdf_path: Path, max_workers: int = 8, check: bool = False
) -> list[str]:
    print(f"Processing {input_pdf_path}...")
    nccn_doc = pymupdf.open(input_pdf_path)
    page_count = nccn_doc.page_count
    nccn_doc.close()

    md_texts = [None] * page_count
    args = [(input_pdf_path, page_id) for page_id in range(page_count)]

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_page, arg): arg[1] for arg in args}
        for future in as_completed(futures):
            page_id, md_text = future.result()
            print(f"Processed page {page_id + 1} of {page_count}.")
            md_texts[page_id] = md_text

    if check:
        for i, md_text in enumerate(md_texts):
            with open("temp.md", "w") as f:
                f.write(md_text)
            os.system("bat temp.md")
            input(f"Press Enter to view page {i + 2}/{len(md_texts)}...")
            os.system("clear")  # Clear screen before showing next page
        if Path("temp.md").exists():
            os.remove("temp.md")  # Remove the temporary file

    return md_texts


if __name__ == "__main__":
    input_pdf_path = config.RAW_DIR / "NCCNGuidelines.pdf"
    output_md_path = config.PROCESSED_DIR / "NCCNGuidelines.md"
    test_pymupdf_nccn(input_pdf_path)
