
import boto3
import sagemaker
import pandas as pd

from pathlib import Path

BUCKET = "mnistmlschool"
S3_LOCATION = f"s3://{BUCKET}/mnist"

sagemaker_client = boto3.client("sagemaker")
iam_client = boto3.client("iam")
role = sagemaker.get_execution_role()
region = boto3.Session().region_name
sagemaker_session = sagemaker.session.Session()
pipeline_session = PipelineSession()
