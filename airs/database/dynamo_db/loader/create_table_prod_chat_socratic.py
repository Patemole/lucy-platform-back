#Création de la table avec global_secondery index pour récupérer l'historique de chat 
import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv
import os

load_dotenv()

AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
AWS_REGION = os.getenv('AWS_REGION')


def create_chat_table():
    # Création d'un client DynamoDB
    dynamodb = boto3.resource(
        'dynamodb',
        region_name=AWS_REGION,  # Assurez-vous que la région est correcte
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY
    )

    try:
        # Création de la table PROD_chat
        table = dynamodb.create_table(
            TableName='PROD_chat_socratic',
            KeySchema=[
                {
                    'AttributeName': 'chat_id',  # Nom de l'attribut pour la clé de partition
                    'KeyType': 'HASH'  # Type de la clé de partition, HASH signifie clé principale
                },
                {
                    'AttributeName': 'message_id',  # Nom de l'attribut pour la clé de tri
                    'KeyType': 'RANGE'  # Type de la clé de tri
                }
            ],
            AttributeDefinitions=[
                {
                    'AttributeName': 'chat_id',
                    'AttributeType': 'S'  # Type 'S' pour String
                },
                {
                    'AttributeName': 'message_id',
                    'AttributeType': 'S'  # Type 'S' pour String
                },
                {
                    'AttributeName': 'course_id',
                    'AttributeType': 'S'  # Type 'S' pour String
                },
                {
                    'AttributeName': 'timestamp',
                    'AttributeType': 'S'  # Type 'S' pour String
                }
            ],
            ProvisionedThroughput={
                'ReadCapacityUnits': 10,
                'WriteCapacityUnits': 10
            },
            GlobalSecondaryIndexes=[
                {
                    'IndexName': 'course_id-index',
                    'KeySchema': [
                        {
                            'AttributeName': 'course_id',
                            'KeyType': 'HASH'
                        },
                        {
                            'AttributeName': 'timestamp',
                            'KeyType': 'RANGE'
                        }
                    ],
                    'Projection': {
                        'ProjectionType': 'ALL'
                    },
                    'ProvisionedThroughput': {
                        'ReadCapacityUnits': 10,
                        'WriteCapacityUnits': 10
                    }
                }
            ]
        )

        # Attendre que la table soit complètement créée
        table.wait_until_exists()
        print("Table created successfully.")
    except ClientError as e:
        error_code = e.response['Error']['Code']
        error_message = e.response['Error']['Message']
        print(f"Error creating table: {error_code} - {error_message}")

# Appel de la fonction pour créer la table
create_chat_table()
