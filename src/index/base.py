from pathlib import Path
from unstructured.chunking.basic import chunk_elements
from unstructured.partition.pdf import partition_pdf
import pymupdf
import pymupdf4llm
from concurrent.futures import ProcessPoolExecutor, as_completed

def get_chunks(
    pdf_file_path: Path,
    max_characters: int = 500,
    overlap: int = 0,
) -> list[str]:
    print(f"Processing {pdf_file_path}...")

    elements = partition_pdf(str(pdf_file_path))
    print(f"Number of elements: {len(elements)}")

    chunks = chunk_elements(elements, max_characters=max_characters, overlap=overlap)
    print(f"Number of chunks: {len(elements)}")

    return [c.text for c in chunks]


def process_page(args: tuple[Path, int]) -> tuple[str, int]:
    input_pdf_path, page_id = args
    nccn_doc = pymupdf.open(input_pdf_path)
    md_text = pymupdf4llm.to_markdown(nccn_doc, pages=[page_id])
    nccn_doc.close()
    return md_text, page_id


def get_md_pages(
    pdf_file_path: Path, output_dir_path: Path, max_workers: int = 8
) -> list[str]:
    output_dir_path.mkdir(parents=True, exist_ok=True)
    print(f"Processing {pdf_file_path}...")
    nccn_doc = pymupdf.open(pdf_file_path)
    page_count = nccn_doc.page_count
    nccn_doc.close()

    md_pages = [None] * page_count
    args = [(pdf_file_path, page_id) for page_id in range(page_count)]

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_page, arg): arg[1] for arg in args}
        for future in as_completed(futures):
            md_text, page_id = future.result()
            print(f"Processed page {page_id + 1} of {page_count}.")
            md_pages[page_id] = md_text

    return md_pages
