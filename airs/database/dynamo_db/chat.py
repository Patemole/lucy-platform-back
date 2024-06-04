from typing import Any, Dict, List
from third_party_api_clients.dynamo_db.dynamo_db_client import DynamoDBClient
from botocore.exceptions import ClientError
from datetime import datetime
import uuid

#table = DynamoDBClient().client.Table("PROD_chat")
table = DynamoDBClient().client.Table("PROD_chat_socratic")

async def get_chat_history(chat_id: str):
    print("\n")
    print("\n")
    print("\n")
    print("\n")
    print(f"Attempting to retrieve chat history for chat_id: {chat_id}")
    try:
        response = table.query(
            KeyConditionExpression='chat_id = :value',
            ExpressionAttributeValues={':value': chat_id}
        )
        items = response.get('Items', [])
        print(f"Retrieved {len(items)} items from chat history.")
        return items
    except ClientError as e:
        error_code = e.response['Error']['Code']
        error_message = e.response['Error']['Message']
        print(f"Error querying chat history: {error_code} - {error_message}")
        return []

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

