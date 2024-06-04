"""
This the handler invoked by the Lambda function that runs whenever there's a change to a hub's uploaded model-output files.
For the Hubverse AWS account, the baseline "hubverse-transform-model-output" Lambda is defined via our IaC (Infrastructure as Code)
repository: hubverse-transform-model-output.

This handler, along with the actual transformation module (hubverse_transform), live outside of the Hubverse's IaC repository:

- To avoid tightly coupling AWS infrastructure to the more general hubverse_transform module that can be used for hubs hosted elsewhere
- To allow faster iteration and testing of the hubverse_transform module without needing to update the IaC repo or redeploy AWS resources
"""
import json
import logging
import urllib.parse

from hubverse_transform.model_output import ModelOutputHandler

logger = logging.getLogger()
logger.setLevel("INFO")


def lambda_handler(event, context):
    logger.info("Received event: " + json.dumps(event, indent=2))

    # info from the S3 event
    event_source = event["Records"][0]["eventSource"]
    event_name = event["Records"][0]["eventName"]
    bucket = event["Records"][0]["s3"]["bucket"]["name"]
    key = urllib.parse.unquote_plus(event["Records"][0]["s3"]["object"]["key"], encoding="utf-8")

    # Below is some old testing code that we were using to ignore all file
    # types that don't have a supported extension. It's commented-out now, to ensure
    # that we don't have to update this handler every time we add support for a  new
    # file type in the model-output transforms.
    # extensions = [".csv", ".parquet"]
    # if not any(ext in key.lower() for ext in extensions):
    #     print(f"{key} is not a supported file type, skipping")
    #     return

    # Until we implement ModelOutputHandler functionality to act on deleted model-output files, ignore any ObjectDelete
    # events that are triggered by by a hub's S3 bucket.
    if "objectcreated" not in event_name.lower():
        logger.info(f"Event type {event_source}:{event_name} is not supported, skipping")
        return

    logger.info(f"Transforming file: {bucket}/{key}")
    try:
        mo = ModelOutputHandler.from_s3(bucket, key)
        mo.transform_model_output()
    except UserWarning:
        pass
    except Exception as e:
        logger.exception(f"Error transforming file: {bucket}/{key}")
        raise e
