import json
import boto3
import os

# Initialize SQS client
sqs = boto3.client("sqs")

# Get SQS queue URL from an environment variable
QUEUE_URL = os.environ.get(
    "SQS_QUEUE_URL"
)  # Set this in your Lambda environment variables


def lambda_handler(event, context):
    # Extract roboflow URL from query string parameters
    url = event.get("queryStringParameters", {}).get("url")
    dtype = event.get("queryStringParameters", {}).get("dataset_type")
    names = event.get("queryStringParameters", {}).get("names")
    names = names.split(",")
    names = json.dumps(names)
    model = event.get("queryStringParameters", {}).get("model")

    if not url:
        return {
            "statusCode": 400,
            "body": json.dumps("url is required as a query string parameter."),
        }

    try:
        # Send message to SQS
        response = sqs.send_message(
            QueueUrl=QUEUE_URL,
            MessageBody=json.dumps(
                {
                    "url": url,
                    "dtype": dtype,
                    "names": names,
                    "model": model,
                }
            ),
        )

        # Return success response with the SQS message ID
        return {
            "statusCode": 200,
            "body": json.dumps(
                {
                    "message": "data sent to SQS",
                    "sqs_message_id": response["MessageId"],
                }
            ),
        }

    except Exception as e:
        # Handle any errors
        return {
            "statusCode": 500,
            "body": json.dumps(f"Error sending message to SQS: {str(e)}"),
        }
