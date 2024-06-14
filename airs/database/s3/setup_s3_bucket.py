import boto3
from botocore.exceptions import ClientError
import logging
from dotenv import load_dotenv
import os

load_dotenv()

AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
AWS_REGION = os.getenv('AWS_REGION')

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("setup_s3_bucket.log")
    ]
)

def create_bucket_if_not_exists(bucket_name: str):
    print(f"Checking if bucket {bucket_name} exists...")
    s3_client = boto3.client(
        's3',
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=AWS_REGION,
    )
    try:
        s3_client.head_bucket(Bucket=bucket_name)
        logging.info(f"Bucket {bucket_name} already exists.")
        print(f"Bucket {bucket_name} already exists.")
    except ClientError as e:
        error_code = int(e.response['Error']['Code'])
        if error_code == 404:
            print(f"Bucket {bucket_name} does not exist. Creating bucket...")
            s3_client.create_bucket(Bucket=bucket_name, CreateBucketConfiguration={'LocationConstraint': AWS_REGION})
            logging.info(f"Bucket {bucket_name} created.")
            print(f"Bucket {bucket_name} created.")
        else:
            logging.error(f"An error occurred while checking/creating the bucket: {e}")
            print(f"An error occurred while checking/creating the bucket: {e}")
            raise e

def create_folders(bucket_name: str):
    s3_client = boto3.client(
        's3',
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=AWS_REGION,
    )
    folders = ['academic_advisor/', 'courses/']
    for folder in folders:
        try:
            print(f"Creating folder {folder} in bucket {bucket_name}...")
            s3_client.put_object(Bucket=bucket_name, Key=folder)
            logging.info(f"Folder {folder} created in bucket {bucket_name}.")
            print(f"Folder {folder} created in bucket {bucket_name}.")
        except ClientError as e:
            logging.error(f"An error occurred while creating folder {folder}: {e}")
            print(f"An error occurred while creating folder {folder}: {e}")
            raise e

def setup_bucket(bucket_name: str):
    create_bucket_if_not_exists(bucket_name)
    create_folders(bucket_name)

if __name__ == "__main__":
    # One bucket per university
    university_bucket = 'usyd-bucket'  # Example bucket name for a university
    setup_bucket(university_bucket)
