import boto3
from pathlib import Path
import time
from loguru import logger
from mypy_boto3_s3.client import S3Client
from mypy_boto3_textract.client import TextractClient
from mypy_boto3_textract.type_defs import GetDocumentAnalysisRequestTypeDef

# ----------------- Configuration -----------------
BUCKET_NAME = "all_files_merged"
LOCAL_PDF_DIR = Path(__file__).parent / "all_files_merged"
OUTPUT_TEXT_DIR = Path(__file__).parent / "textract_output"
REGION_NAME = "us-east-1"


def upload_to_s3(s3: S3Client, file_path: Path, bucket: str, object_name: str):
    try:
        s3.upload_file(str(file_path), bucket, object_name)
        logger.info(f"✅ Uploaded: {object_name}\n")
    except Exception:
        logger.exception(f"❌ Failed to upload {object_name}")


def start_textract_job(textract: TextractClient, bucket: str, document: str) -> str:
    try:
        response = textract.start_document_text_detection(
            DocumentLocation={"S3Object": {"Bucket": bucket, "Name": document}}
        )
        job_id = response["JobId"]
        logger.info(f"🚀 Started Textract job {job_id} for {document}\n")
        return job_id
    except Exception:
        logger.exception(f"❌ Failed to start Textract job for {document}")
        raise


def is_job_complete(textract: TextractClient, job_id: str) -> bool:
    try:
        while True:
            response = textract.get_document_text_detection(JobId=job_id)
            status = response["JobStatus"]
            logger.debug(f"⏳ Job {job_id} status: {status}")
            if status in ["SUCCEEDED", "FAILED"]:
                return status == "SUCCEEDED"
            time.sleep(5)
    except Exception:
        logger.exception(f"❌ Error while polling job {job_id}")
        raise


def get_job_results(textract: TextractClient, job_id: str) -> list[dict]:
    pages = []
    next_token = None
    try:
        while True:
            kwargs: GetDocumentAnalysisRequestTypeDef = {"JobId": job_id}
            if next_token:
                kwargs["NextToken"] = next_token

            response = textract.get_document_text_detection(**kwargs)
            pages.extend(response["Blocks"])
            next_token = response.get("NextToken")
            if not next_token:
                break
        logger.info(f"📄 Retrieved {len(pages)} blocks for job {job_id}\n")
        return pages
    except Exception:
        logger.exception(f"❌ Failed to retrieve results for job {job_id}")
        raise


def extract_text_from_blocks(blocks: list[dict]) -> str:
    lines = [block["Text"] for block in blocks if block["BlockType"] == "LINE"]
    return "\n".join(lines)


def process_pdf_file(s3: S3Client, textract: TextractClient, file_path: Path):
    filename = file_path.name
    try:
        logger.info(f"📥 Processing file: {filename}\n")
        upload_to_s3(s3, file_path, BUCKET_NAME, filename)
        job_id = start_textract_job(textract, BUCKET_NAME, filename)
        if is_job_complete(textract, job_id):
            blocks = get_job_results(textract, job_id)
            text = extract_text_from_blocks(blocks)
            OUTPUT_TEXT_DIR.mkdir(parents=True, exist_ok=True)
            output_file = OUTPUT_TEXT_DIR / f"{filename}.txt"
            output_file.write_text(text)
            logger.success(f"✅ Text written to {output_file}\n")
        else:
            logger.error(f"❌ Textract job {job_id} failed for {filename}")
    except Exception:
        logger.exception(f"❌ Unexpected error while processing {filename}")


def process():
    s3 = boto3.client("s3", region_name=REGION_NAME)
    textract = boto3.client("textract", region_name=REGION_NAME)
    if not LOCAL_PDF_DIR.is_dir():
        raise ValueError(f"{LOCAL_PDF_DIR} is not a directory")
    for pdf_path in LOCAL_PDF_DIR.iterdir():
        if pdf_path.suffix.lower() == ".pdf":
            process_pdf_file(s3, textract, pdf_path)


if __name__ == "__main__":
    process()
