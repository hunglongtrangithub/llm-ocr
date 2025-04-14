from moto import mock_aws
from ocr import process, BUCKET_NAME, REGION_NAME
from loguru import logger
import boto3
import os


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
