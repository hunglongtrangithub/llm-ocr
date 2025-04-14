from moto import mock_aws
from ocr import process, REGION_NAME, BUCKET_NAME
import boto3


@mock_aws
def test_process():
    s3 = boto3.client("s3", region_name=REGION_NAME)
    # Create fake S3 bucket
    s3.create_bucket(Bucket=BUCKET_NAME)
    # Add fake PDFs, or skip upload if testing only pipeline
    process()


if __name__ == "__main__":
    test_process()  # or main() for real AWS
