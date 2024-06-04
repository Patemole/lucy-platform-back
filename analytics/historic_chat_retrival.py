import logging
import sys
import os
import boto3
from datetime import datetime, timedelta
from botocore.exceptions import ClientError

# Classe DynamoDBClient
class DynamoDBClient:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            print("Creating new instance of DynamoDBClient_greg_test")
            try:
                cls._instance = super(DynamoDBClient, cls).__new__(cls)
                cls.client = boto3.resource(
                    'dynamodb',
                    aws_access_key_id='AKIA2UC3A5LOLSDOW6X7',
                    aws_secret_access_key='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',
                    region_name='ap-southeast-2'
                )
                print("DynamoDBClient initialized successfully")
            except Exception as e:
                print(f"Failed to initialize DynamoDBClient: {e}")
        else:
            print("Using existing instance of DynamoDBClient")
        
        return cls._instance

# Configure logging to print messages to the console
logging.basicConfig(level=logging.INFO)

# Définir la table DynamoDB
table = DynamoDBClient().client.Table("PROD_chat_socratic")

# Définir la fonction pour récupérer les messages de chat d'un cours spécifique
def get_course_chat_messages(course_id: str, start_date: datetime, end_date: datetime = datetime.now()):
    try:
        # Query chats for a specific course using the CourseIndex GSI
        response = table.query(
            IndexName='course_id-index',
            KeyConditionExpression='course_id = :course_id AND #ts BETWEEN :start_date AND :end_date',
            FilterExpression='username <> :tai_username',
            ExpressionAttributeValues={
                ':course_id': course_id,
                ':start_date': start_date.isoformat(),
                ':end_date': end_date.isoformat(),
                ':tai_username': "TAI"
            },
            ExpressionAttributeNames={
                "#ts": "timestamp"
            },
            ScanIndexForward=True  # Set to True for chronological order
        )
        return response.get('Items', [])
    except ClientError as e:
        error_code = e.response['Error']['Code']
        error_message = e.response['Error']['Message']
        print(f"Error querying chats for course with ID {course_id}: {error_code} - {error_message}")
        return []

# Fonction de test
def test_get_course_chat_messages():
    # Définir les paramètres de test
    course_id = "course_id_test"
    start_date = datetime.now() - timedelta(days=200)
    end_date = datetime.now()  # Optionnel, vous pouvez omettre pour utiliser la valeur par défaut

    # Appeler la fonction et afficher les résultats
    try:
        messages = get_course_chat_messages(course_id, start_date, end_date)
        print(f"Nombre de messages reçus pour le cours {course_id}: {len(messages)}")
        for message in messages:
            print(message)
    except Exception as e:
        print(f"Erreur lors du test de la fonction: {e}")

if __name__ == "__main__":
    test_get_course_chat_messages()
