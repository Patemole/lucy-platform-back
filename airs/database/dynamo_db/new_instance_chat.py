from typing import Any, Dict, List
from botocore.exceptions import ClientError
from datetime import datetime
import uuid
import boto3
from dotenv import load_dotenv
import os

load_dotenv()

AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
AWS_REGION = os.getenv('AWS_REGION')

# Création du client DynamoDB
dynamodb = boto3.resource(
    'dynamodb',
    region_name=AWS_REGION,  # Assurez-vous que la région est correcte
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY
)

# Référence à la table
table = dynamodb.Table('PROD_chat_socratic')

async def delete_all_items_and_adding_first_message(chat_id: str):
    try:
        # Scan the table to find all items
        response = table.scan()
        items = response['Items']

        # Delete each item
        with table.batch_writer() as batch:
            for item in items:
                batch.delete_item(Key={'message_id': item['message_id'], 'chat_id': item['chat_id']})

        print("All items in the table deleted successfully.")

        # Add a new initial message with the given chat_id
        await store_message_async(chat_id, 'no_course', 'Initial message after deletion', 'TAI')

        print(f"Initial message for chat_id {chat_id} created successfully.")

    except ClientError as e:
        error_code = e.response['Error']['Code']
        error_message = e.response['Error']['Message']
        print(f"Error deleting chat history: {error_code} - {error_message}")
        raise


async def store_message_async(
        chat_id: str, 
        course_id: str, 
        message_body: str, 
        username: str = "TAI",
        documents: List[Dict[str, Any]] = []):
    print(f"Attempting to store message for chat_id: {chat_id}, course_id: {course_id}")
    try:
        # Insert the item into DynamoDB
        args = {
            'message_id': str(uuid.uuid4()),
            'chat_id': chat_id,
            'timestamp': datetime.now().isoformat(),
            'course_id': course_id,
            'body': message_body,
            'username': username
        }
        if username == "TAI" and documents:
            args['documents'] = documents
        table.put_item(Item=args)
        print(f"Message stored successfully with message_id: {args['message_id']}")
    except ClientError as e:
        error_code = e.response['Error']['Code']
        error_message = e.response['Error']['Message']
        print(f"Error inserting message into chat history: {error_code} - {error_message}")
