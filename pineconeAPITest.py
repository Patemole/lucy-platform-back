
#To be depreciated
import logging
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
import langchain_pinecone
import os
from pinecone import Pinecone, Index
import langchain
from typing import List
from langchain.schema import Document
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
PINECONE_POOL_THREADS = 4

class OpenAIApiClient:
    _instance = None

    def __init__(self):
        self.text_embeddings = OpenAIEmbeddings(
            openai_api_key=OPENAI_API_KEY,
            model="text-embedding-3-small"
        )
        self.open_ai_client = OpenAI(api_key=OPENAI_API_KEY)
        
    def get_langchain_open_ai_api_client(self):
        return langchain.OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)

class PineconeApiClient:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(PineconeApiClient, cls).__new__(cls)
            # Initialize the Pinecone API client
            pinecone = Pinecone(api_key=PINECONE_API_KEY, environment="gcp-starter")
            # Initialize Index
            cls.index: Index = pinecone.Index("academic-advisor-upenn-test1", pool_threads=PINECONE_POOL_THREADS)
            # Get OpenAI client for embeddings
            text_embeddings = OpenAIApiClient().text_embeddings
            # Initialize Vector Store
            cls.vectorstore = langchain_pinecone.Pinecone(cls.index, text_embeddings, "text")
        return cls._instance

def generate_enhanced_query(user_input, keywords):
    combined_query = user_input + " " + " ".join(keywords)
    return combined_query

def query_pinecone(user_input, class_id, keywords):
    # Generate the enhanced query using the user input and keywords
    query = generate_enhanced_query(user_input, keywords)

    # Define the metadata filter
    filter = {"class_id": class_id}

    # Retrieve documents
    #retrieved_docs: List[Document] = PineconeApiClient().vectorstore.similarity_search(query=query, k=3, filter=filter)
    retrieved_docs: List[Document] = PineconeApiClient().vectorstore.similarity_search(query=query, k=3)
    
    return retrieved_docs

# Example usage
if __name__ == "__main__":
    client = PineconeApiClient()
    print("PineconeApiClient initialized successfully.")
    
    # Test retrieval
    user_input = "What courses for computer science?"
    class_id = "csi-1010-introduction-to-economics-vrj2px"
    keywords = ["principles", "economics"]
    
    results = query_pinecone(user_input, class_id, keywords)
    for result in results:
        print(result)
