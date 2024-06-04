from langchain_openai import OpenAIEmbeddings
import os
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

class OpenAIApiClient:
    _instance = None
 
    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(OpenAIApiClient, cls).__new__(cls)
            cls.text_embeddings = OpenAIEmbeddings(
                #openai_api_key="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",
                openai_api_key = OPENAI_API_KEY,
                model="text-embedding-ada-002"
            )
            cls.open_ai_client = OpenAI(api_key=OPENAI_API_KEY)
            

        return cls._instance