from pathlib import Path

import pymupdf
from loguru import logger


def process_pdf_file(pdf_path: Path, txt_path: Path):
    report = pymupdf.open(str(pdf_path))
    txt_file = open(txt_path, "wb")
    for page in report.pages():
        text = page.get_text().encode("utf8")
        txt_file.write(text)
        txt_file.write(bytes((12,)))
    txt_file.close()


def process(in_pdf_dir: Path, out_txt_dir: Path):
    if not in_pdf_dir.is_dir():
        raise ValueError(f"{in_pdf_dir} is not a directory")
    total_count = 0
    for pdf_path in in_pdf_dir.iterdir():
        if pdf_path.suffix.lower() == ".pdf":
            total_count += 1
            report_name = pdf_path.stem
            txt_path = out_txt_dir / f"{report_name}.txt"
            logger.info(f"Processing report {report_name}")
            process_pdf_file(pdf_path, txt_path)
    logger.info(f"Processed {total_count} files")


if __name__ == "__main__":
    from src import config

    in_pdf_dir = config.RAW_DIR / "TCGA_Reports_pdf"
    out_txt_dir = config.PROCESSED_DIR / "TCGA_Reports_txt"
    process(in_pdf_dir, out_txt_dir)
