import os
import sys
import logging

from openai_api_client import OpenAIApiClient
from config import PINECONE_POOL_THREADS
import langchain_pinecone
from pinecone import Pinecone, Index
from dotenv import load_dotenv
import os

load_dotenv()

PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')


logging.basicConfig(level=logging.INFO)


class PineconeApiClient:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            logging.info("Initialisation de l'instance PineconeApiClient...")
            cls._instance = super(PineconeApiClient, cls).__new__(cls)
            try:
                pinecone = Pinecone(PINECONE_API_KEY)
                cls.index = pinecone.Index("adding-files-socratic-assistant", pool_threads=PINECONE_POOL_THREADS)
                text_embeddings = OpenAIApiClient().text_embeddings
                cls.vectorstore = langchain_pinecone.Pinecone(cls.index, text_embeddings, "text")
                logging.info("PineconeApiClient a été initialisé avec succès _test_greg.")
            except Exception as e:
                logging.error(f"Erreur lors de l'initialisation de PineconeApiClient: {e}")
                raise e
        return cls._instance