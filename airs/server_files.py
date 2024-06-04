import sys
import os
import asyncio
import logging
from fastapi import APIRouter, FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any, List, Tuple
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader
from pdf2image import convert_from_path
import langchain_pinecone
from pinecone import Pinecone
import boto3
from botocore.exceptions import ClientError
from datetime import datetime
from uuid import uuid4
from pinecone import Pinecone, ServerlessSpec
import io
import requests
import tempfile
from dotenv import load_dotenv



PINECONE_POOL_THREADS = 4
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 50
PINECONE_BATCH_SIZE = 64

# Endpoint for creating the Pinecone index
class IndexRequest(BaseModel):
    pinecone_index_name: str



# Add necessary directories to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("file_server.log")
    ]
)

app = FastAPI(
    title="File Service",
    version="0.0.1"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

file_router = APIRouter(prefix='/file', tags=['file'])



##################################AWS CONFIGURATION############################################
load_dotenv()

#AWS
AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
AWS_REGION = os.getenv('AWS_REGION')

#Pinecone
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')

#OpenAI
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

'''
AWS_ACCESS_KEY_ID = 'AKIA2UC3A5LOLSDOW6X7'
AWS_SECRET_ACCESS_KEY = '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'
AWS_REGION = 'ap-southeast-2'
'''

s3_client = boto3.client(
    's3',
    region_name=AWS_REGION,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY
)

dynamodb = boto3.resource(
    'dynamodb',
    region_name=AWS_REGION,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY
)
##################################AWS CONFIGURATION############################################







##################################API ROUTES############################################
#1
@file_router.post("/api/create-pinecone-index")
async def create_pinecone_index(request: IndexRequest):
    logging.info("Received request to create Pinecone index: %s", request.pinecone_index_name)
    print(f"Received request to create Pinecone index: {request.pinecone_index_name}")

    try:
        #pc = Pinecone(api_key="5a6353bd-2e4a-436a-b6f4-8a18187884e1")
        pc = Pinecone(api_key=PINECONE_API_KEY)

        pc.create_index(
            name=request.pinecone_index_name,
            dimension=3072,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )

        logging.info("Successfully created Pinecone index: %s", request.pinecone_index_name)
        print(f"Successfully created Pinecone index: {request.pinecone_index_name}")

        return {"message": f"Successfully created Pinecone index: {request.pinecone_index_name}"}
    except Exception as e:
        logging.error("Failed to create Pinecone index: %s", e)
        print(f"Failed to create Pinecone index: {e}")
        raise HTTPException(status_code=500, detail=str(e))


#2
@file_router.post("/upload_file_from_teacher_dashboard")
async def upload_file(
    file: UploadFile = File(...),
    uid: str = Form(...),
    course_id: str = Form(...),
    pinecone_index_name: str = Form(...),
    bucket_name: str = Form(...)
):
    logging.info(f"Arrivé dans le endpoint : Traitement du fichier {file.filename} pour l'utilisateur {uid} et le cours {course_id}")
    logging.info(f"Voici l'intérieur de l'object {file}")
    try:
        # Process embeddings and get the S3 URL
        s3_url, preview_url = await embeddings_treatment(file, uid, course_id, pinecone_index_name, bucket_name, None)

        file_id = str(uuid4()) # Generate a unique file ID
        save_file_metadata_to_dynamodb(file_id, file.filename, course_id, uid, file.filename, s3_url, preview_url)
        
        return {"file_name": file.filename, "preview_url": preview_url}
    except Exception as e:
        logging.error(f"Erreur lors du traitement du fichier : {str(e)}")
        raise HTTPException(status_code=500, detail="Erreur lors du traitement du fichier")


#3
@file_router.get("/fetching/{course_id}")
async def fetch_files(course_id: str):
    print(f"Fetching files for course_id: {course_id}")
    logging.info(f"Fetching files for course_id: {course_id}")
    try:
        table = dynamodb.Table('PROD_files_teacher_dashboard')
        response = table.query(
            IndexName='course_id-index',
            KeyConditionExpression=boto3.dynamodb.conditions.Key('course_id').eq(course_id)
        )
        files = response.get('Items', [])
        if not files:
            print("No files found")
            logging.info("No files found for the given course ID")
            return {"message": "No files found for the given course ID"}
        
        result = [
            {"name": file["name"], "preview_url": file.get("url_image_preview", None)}
            for file in files
        ]
        
        print(f"Files fetched: {result}")
        logging.info(f"Files fetched for course_id {course_id}: {result}")
        return {"files": result}
    except Exception as e:
        logging.error(f"Error fetching files for course_id {course_id}: {str(e)}")
        print(f"Error fetching files for course_id {course_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Error fetching files")

##################################END API ROUTES############################################




def save_file_metadata_to_dynamodb(file_id, name, course_id, uid, file_type, url, preview_url=None):
    table = dynamodb.Table('PROD_files_teacher_dashboard')
    timestamp = datetime.utcnow().isoformat()

    try:
        item = {
            'file_id': file_id,
            'name': name,
            'course_id': course_id,
            'uid': uid,
            'type': file_type,
            'url': url,
            'created_at': timestamp,
            'updated_at': timestamp,
        }
        if preview_url:
            item['url_image_preview'] = preview_url

        table.put_item(Item=item)
        print(f"File metadata for {name} saved to DynamoDB.")
    except ClientError as e:
        print(f"Failed to save file metadata to DynamoDB: {e.response['Error']['Message']}")




def generate_pdf_preview(file_path, bucket_name, file_id, file_name, course_id):
    # Convert PDF to image
    images = convert_from_path(file_path, first_page=1, last_page=1)
    preview_image_path = f"/tmp/{file_name}-preview-image.jpg"
    images[0].save(preview_image_path, 'JPEG')
    
    # Upload preview image to S3
    s3_client = boto3.client('s3', aws_access_key_id=AWS_ACCESS_KEY_ID, aws_secret_access_key=AWS_SECRET_ACCESS_KEY, region_name=AWS_REGION)
    preview_key = f"courses/{course_id}/previews/{file_name}-preview-image.jpg"
    s3_client.upload_file(preview_image_path, bucket_name, preview_key)
    
    preview_url = f"https://{bucket_name}.s3.{AWS_REGION}.amazonaws.com/{preview_key}"
    return preview_url



def save_uploaded_file(uploaded_file: UploadFile) -> str:
    try:
        if not os.path.exists("uploaded_files"):
            os.mkdir("uploaded_files")
        file_path = os.path.join("uploaded_files", uploaded_file.filename)
        
        with open(file_path, "wb") as f:
            content = uploaded_file.file.read()
            f.write(content)
        
        # Check if the file was written correctly
        if os.path.getsize(file_path) == 0:
            raise Exception("File size is 0 bytes after saving.")
        
        logging.info(f"File saved successfully at: {file_path} with size {os.path.getsize(file_path)} bytes")
        return file_path
    except Exception as e:
        logging.error(f"An error occurred while saving the file locally: {e}")
        return None

def save_file_to_s3(uploaded_file: UploadFile, file_path: str, bucket_name: str, course_id: str) -> str:
    try:
        s3_client = boto3.client(
            's3',
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            region_name=AWS_REGION,
        )

        # Check if the course folder exists
        course_prefix = f'courses/{course_id}/'
        result = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=course_prefix)
        if 'Contents' not in result:
            print(f"Creating new course folder: {course_prefix}")
            s3_client.put_object(Bucket=bucket_name, Key=course_prefix)
        
        # Save the file in the course folder
        s3_client.upload_file(file_path, bucket_name, course_prefix + uploaded_file.filename)

        s3_url = f"https://{bucket_name}.s3.{AWS_REGION}.amazonaws.com/{course_prefix}{uploaded_file.filename}"
        return s3_url
    except Exception as e:
        print(f"An error occurred while uploading to S3: {e}")
        return None

async def embeddings_treatment(uploaded_file: UploadFile, uid: str, course_id: str, pinecone_index_name: str, bucket_name: str, s3_url: str) -> str:
    print("Starting embeddings_treatment")

    class Chunk:
        def __init__(self, text: str, metadata: Dict[str, Any]):
            self.text = text
            self.metadata = metadata

        def update_metadata(self, key: str, value: Any):
            self.metadata[key] = value

        def remove_metadata(self, key: str):
            if key in self.metadata:
                del self.metadata[key]

    class ParsedChunksFile:
        def __init__(self, filename: str):
            self.filename = filename
            self.chunks = []

        def get_filename(self):
            return self.filename
        
        def add_chunk(self, chunk: Chunk):
            self.chunks.append(chunk)

        def add_chunks(self, chunks: List[Chunk]):
            self.chunks.extend(chunks)

        def get_file_metadata(self) -> Dict[str, Any]:
            return {"filename": self.filename}

        def get_chunks(self) -> List[Chunk]:
            return self.chunks

        def get_chunks_with_metadata(self) -> Tuple[List[Dict[str, Any]], List[str]]:
            return [chunk.metadata for chunk in self.chunks], [chunk.text for chunk in self.chunks]

    class SentenceAwareTextSplitter:
        def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200, separators: List[str] = None):
            self.chunk_size = chunk_size
            self.chunk_overlap = chunk_overlap
    
        def split_text(self, text: str) -> List[str]:
            sentences = text.split('. ')
            chunks = []
            current_chunk = ""
        
            for sentence in sentences:
                if len(current_chunk) + len(sentence) + 1 > 1000:
                    if current_chunk:
                        chunks.append(current_chunk)
                    current_chunk = sentence
                else:
                    if current_chunk:
                        current_chunk += ". " + sentence
                    else:
                        current_chunk = sentence
        
            if current_chunk:
                chunks.append(current_chunk)
        
            # Apply chunk overlap
            return chunks

    class OpenAIApiClient:
        _instance = None

        def __new__(cls, *args, **kwargs):
            if not cls._instance:
                cls._instance = super(OpenAIApiClient, cls).__new__(cls)
                cls.text_embeddings = OpenAIEmbeddings(
                    openai_api_key=OPENAI_API_KEY,
                    model="text-embedding-3-large"
                )
                cls.open_ai_client = OpenAI(api_key=OPENAI_API_KEY)
            return cls._instance

    class PineconeApiClient:
        _instance = None

        def __new__(cls, *args, **kwargs):
            if not cls._instance:
                logging.info("Initialisation de l'instance PineconeApiClient en test pour le socratic...")
                cls._instance = super(PineconeApiClient, cls).__new__(cls)
                try:
                    pinecone = Pinecone(api_key=PINECONE_API_KEY)

                    # Utiliser le nom de l'index passé dans l'endpoint
                    index_name = pinecone_index_name

                    logging.info(f"Using index: {index_name}")

                    cls.index = pinecone.Index(index_name, pool_threads=4)
                    text_embeddings = OpenAIApiClient().text_embeddings
                    cls.vectorstore = langchain_pinecone.Pinecone(cls.index, text_embeddings, "text")
                    logging.info("PineconeApiClient a été initialisé avec succès.")
                except Exception as e:
                    logging.error(f"Erreur lors de l'initialisation de PineconeApiClient: {e}")
                    raise e
            return cls._instance

    async def store_parsed_chunks(
        parsed_chunks_file: ParsedChunksFile,
        namespace: str = ""
    ):
        metadatas, chunks = parsed_chunks_file.get_chunks_with_metadata()
        total_vectors_stored = 0
        # Store Chunks
        i = 0
        while i < len(chunks):
            vectors_stored = PineconeApiClient().vectorstore.add_texts(
                texts=chunks[i: i + 64],
                metadatas=metadatas[i: i + 64],
                namespace=namespace,
                batch_size=64
            )
            total_vectors_stored += len(vectors_stored)
            i += 64

        logging.info(f"Stored {total_vectors_stored} vectors from file {parsed_chunks_file.get_filename()} into namespace {namespace}")
        return total_vectors_stored

    async def pdf_parser(file_obj, metadata: dict = {}):
        logging.info("Parsing pdf from file object")
        pdf_reader = PdfReader(file_obj)
        content = ""

        for page_num, page in enumerate(pdf_reader.pages):
            text = page.extract_text()
            logging.info(f"Page {page_num} text length: {len(text) if text else 'None'}")
            if text:
                content += text
            else:
                logging.warning(f"Page {page_num} has no text")

        if not content:
            raise Exception("The file content is empty after extraction")

        text_splitter = SentenceAwareTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )

        chunks = text_splitter.split_text(content)
        
        parsed_file = ParsedChunksFile(filename=uploaded_file.filename)
        parsed_file.add_chunks([Chunk(metadata={"filename": uploaded_file.filename, **metadata}, text=chunk) for chunk in chunks])

        return parsed_file, content

    # Upload the file to S3
    s3_client = boto3.client('s3')
    s3_key = f"courses/{course_id}/{uploaded_file.filename}"
    s3_client.upload_fileobj(uploaded_file.file, bucket_name, s3_key)
    s3_url = f"https://{bucket_name}.s3.amazonaws.com/{s3_key}"
    print(f"File saved to S3 at: {s3_url}")

    # Generate PDF preview if the file is a PDF
    preview_url = None
    if uploaded_file.filename.lower().endswith('.pdf'):
        response = requests.get(s3_url)
        file_obj = io.BytesIO(response.content)

        # Save BytesIO content to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            temp_file.write(file_obj.getbuffer())
            temp_file_path = temp_file.name

        try:
            preview_url = generate_pdf_preview(temp_file_path, bucket_name, str(uuid4()), os.path.splitext(uploaded_file.filename)[0], course_id)
        finally:
            # Clean up temporary file
            os.remove(temp_file_path)

    # Prepare metadata for embeddings
    file_extension = os.path.splitext(uploaded_file.filename)[1].lower()
    metadata = {'course_id': course_id, 'resource_name': uploaded_file.filename, 'resource_link': s3_url}

    if file_extension == '.pdf':
        response = requests.get(s3_url)
        file_obj = io.BytesIO(response.content)
        parsed_chunks_file, content = await pdf_parser(file_obj, metadata)
        if not content:
            raise Exception("Parsed content is empty")
        num_vectors_uploaded = await store_parsed_chunks(parsed_chunks_file)
        print(f"Number of vectors uploaded for {uploaded_file.filename}: {num_vectors_uploaded}")
    elif file_extension == '.mp4':
        print(f"Received an MP4 file {uploaded_file.filename}. No parsing needed.")
    else:
        raise Exception(f"Unsupported file extension: {file_extension}")

    return s3_url, preview_url

app.include_router(file_router)

def run():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8005)

if __name__ == "__main__":
    run()
