import os
import time
from pathlib import Path

import boto3
from loguru import logger
from moto import mock_aws
from mypy_boto3_s3.client import S3Client
from mypy_boto3_textract.client import TextractClient
from mypy_boto3_textract.type_defs import GetDocumentAnalysisRequestTypeDef

from config import PROCESSED_DIR, RAW_DIR

# ----------------- Configuration -----------------
PDF_DIR = RAW_DIR / "TCGA_Reports_pdf"
TXT_DIR = PROCESSED_DIR / "TCGA_Reports_txt"
REGION_NAME = "us-east-1"
BUCKET_NAME = "TCGA_Reports_pdf"


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


def process_pdf_file(s3: S3Client, textract: TextractClient, file_path: Path) -> bool:
    filename = file_path.name
    try:
        logger.info(f"📥 Processing file: {filename}\n")
        upload_to_s3(s3, file_path, BUCKET_NAME, filename)
        job_id = start_textract_job(textract, BUCKET_NAME, filename)
        if is_job_complete(textract, job_id):
            blocks = get_job_results(textract, job_id)
            text = extract_text_from_blocks(blocks)
            TXT_DIR.mkdir(parents=True, exist_ok=True)
            output_file = TXT_DIR / f"{filename}.txt"
            output_file.write_text(text)
            logger.success(f"✅ Text written to {output_file}\n")
            return True
        else:
            logger.error(f"❌ Textract job {job_id} failed for {filename}")
            return False
    except Exception:
        logger.exception(f"❌ Unexpected error while processing {filename}")
        return False


def process():
    s3 = boto3.client("s3", region_name=REGION_NAME)
    textract = boto3.client("textract", region_name=REGION_NAME)
    if not PDF_DIR.is_dir():
        raise ValueError(f"{PDF_DIR} is not a directory")
    total_count = 0
    success_count = 0
    for pdf_path in PDF_DIR.iterdir():
        if pdf_path.suffix.lower() == ".pdf":
            total_count += 1
            if process_pdf_file(s3, textract, pdf_path):
                success_count += 1
    logger.info(f"Successfully processed {success_count}/{total_count} files")


# Set up testing environment for AWS S3
def setup_testing_s3():
    bucket_name = BUCKET_NAME
    # Set up fake AWS credentials
    os.environ["AWS_ACCESS_KEY_ID"] = "testing"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "testing"
    os.environ["AWS_SECURITY_TOKEN"] = "testing"
    os.environ["AWS_SESSION_TOKEN"] = "testing"
    os.environ["AWS_DEFAULT_REGION"] = REGION_NAME
    os.environ["S3_BUCKET_NAME"] = bucket_name

    # Mock S3 and create a test bucket with objects
    def setup_mock_s3():
        s3 = boto3.resource("s3")
        s3.create_bucket(Bucket=bucket_name)
        logger.info(f"Mock S3 bucket '{bucket_name}' created.")

    # Execute the mock setup
    setup_mock_s3()


def test_process():
    with mock_aws():
        setup_testing_s3()
        process()


if __name__ == "__main__":
    test_process()  # or process() for real AWS
