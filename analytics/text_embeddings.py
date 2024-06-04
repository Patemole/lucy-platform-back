from langchain_openai import OpenAIEmbeddings
import os
from openai import OpenAI


#################################
class OpenAIApiClient:
    _instance = None
 
    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(OpenAIApiClient, cls).__new__(cls)
            cls.text_embeddings = OpenAIEmbeddings(
                #openai_api_key="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",
                openai_api_key ="sk-proj-EbsswciW1QbyS50aDohkT3BlbkFJgawGIcLqmGMMNjCEe00M",
                model="text-embedding-ada-002"
            )
            cls.open_ai_client = OpenAI(api_key="sk-proj-EbsswciW1QbyS50aDohkT3BlbkFJgawGIcLqmGMMNjCEe00M")
            

        return cls._instance

#################################
def get_text_embedding(text, model="text-embedding-3-small"):   
   client = OpenAIApiClient().open_ai_client
   text = text.replace("\n", " ")
   return client.embeddings.create(input = [text], model=model).data[0].embedding