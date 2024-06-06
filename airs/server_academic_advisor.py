'''
import sys
import os
import asyncio
import logging
from fastapi import APIRouter, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import asyncio
from fastapi import APIRouter, FastAPI, Request, Response, UploadFile, HTTPException, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from airs.database.dynamo_db.chat import get_chat_history, store_message_async
from dotenv import load_dotenv
from airs.model.input_query import InputQuery
from airs.database.dynamo_db.new_instance_chat import delete_all_items_and_adding_first_message
from airs.academic_advisor import academic_advisor_answer_generation


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


############################################################
# Environment variables
############################################################
load_dotenv()

#AWS
AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
AWS_REGION = os.getenv('AWS_REGION')

#Pinecone
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')

#OpenAI
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')


############################################################
# FastAPI app configuration
############################################################
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

chat_router = APIRouter(prefix='/chat', tags=['chat'])




############################################################
# TRAITEMENT D'UN MESSAGE ÉLÈVE AVEC LA LOGIQUE LANGRAPH
############################################################
@chat_router.post("/send_message_socratic_langgraph")
async def chat(request: Request, response: Response, input_query: InputQuery) -> StreamingResponse:
    chat_id = input_query.chat_id
    course_id = input_query.course_id  # Get course_id from input_query
    
    # Récupérez l'historique de chat
    chat_history = await get_chat_history(chat_id)
    print(chat_history)
    
    # Stockez le message de manière asynchrone
    asyncio.ensure_future(store_message_async(chat_id, username=input_query.username, course_id=input_query.course_id, message_body=input_query.message))

    # Créez une réponse en streaming en passant l'historique de chat
    return StreamingResponse(academic_advisor_answer_generation(input_query.message, chat_history), media_type="text/event-stream")

    #Il faudra appeler une autre fonction en fonction de si c'est l'academic advisor ou d'autres cours. 
    #Logic suivante : if course_id == "academic_advisor" : return StreamingResponse(academic_advisor_answer_generation(input_query.message, chat_history), media_type="text/event-stream")
    #else : return StreamingResponse(socratic_answer_generation(input_query.message, chat_history, course_id), media_type="text/event-stream") 
    #avec le course_id qui correspond  au cours que l'élève a demandé. 




############################################################
#RÉCUPÉRATION DE L'HISTORIQUE DE CHAT POUR LANGRAPH
############################################################
@chat_router.get("/get_chat_history/{chat_id}")
async def get_chat_history_route(chat_id: str):
    return await get_chat_history(chat_id)



############################################################
#SUPPRIMER L'HISTORIQUE DE CHAT  CHAQUE CHARGEMENT DE LA PAGE - TO BE DEPRECIATED
############################################################
@chat_router.post("/delete_chat_history/{chat_id}")
async def delete_chat_history_route(chat_id: str):
    try:
        await delete_all_items_and_adding_first_message(chat_id)
        return {"message": "Chat history deleted successfully"}
    except Exception as e:
        logging.error(f"Erreur lors de la suppression de l'historique du chat : {str(e)}")
        raise HTTPException(status_code=500, detail="Erreur lors de la suppression de l'historique du chat")
'''

import sys
import os
import asyncio
import logging
from fastapi import APIRouter, FastAPI, HTTPException, Request, Response, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from airs.database.dynamo_db.chat import get_chat_history, store_message_async
from airs.model.input_query import InputQuery
from airs.database.dynamo_db.new_instance_chat import delete_all_items_and_adding_first_message
from airs.academic_advisor import academic_advisor_answer_generation

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

# Environment variables
load_dotenv()

# AWS
AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
AWS_REGION = os.getenv('AWS_REGION')

# Pinecone
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')

# OpenAI
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# FastAPI app configuration
app = FastAPI(
    title="Chat Service",
    version="0.0.1"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#chat_router = APIRouter(prefix='/chat', tags=['chat'])

# TRAITEMENT D'UN MESSAGE ÉLÈVE AVEC LA LOGIQUE LANGRAPH
@app.post("/send_message_socratic_langgraph")
async def chat(request: Request, response: Response, input_query: InputQuery) -> StreamingResponse:
    chat_id = input_query.chat_id
    course_id = input_query.course_id  # Get course_id from input_query
    
    # Récupérez l'historique de chat
    chat_history = await get_chat_history(chat_id)
    print(chat_history)
    
    # Stockez le message de manière asynchrone
    asyncio.ensure_future(store_message_async(chat_id, username=input_query.username, course_id=input_query.course_id, message_body=input_query.message))

    # Créez une réponse en streaming en passant l'historique de chat
    return StreamingResponse(academic_advisor_answer_generation(input_query.message, chat_history), media_type="text/event-stream")

    # Il faudra appeler une autre fonction en fonction de si c'est l'academic advisor ou d'autres cours.
    # Logic suivante : if course_id == "academic_advisor" : return StreamingResponse(academic_advisor_answer_generation(input_query.message, chat_history), media_type="text/event-stream")
    # else : return StreamingResponse(socratic_answer_generation(input_query.message, chat_history, course_id), media_type="text/event-stream")
    # avec le course_id qui correspond au cours que l'élève a demandé.

# RÉCUPÉRATION DE L'HISTORIQUE DE CHAT POUR LANGRAPH
@app.get("/get_chat_history/{chat_id}")
async def get_chat_history_route(chat_id: str):
    return await get_chat_history(chat_id)

# SUPPRIMER L'HISTORIQUE DE CHAT CHAQUE CHARGEMENT DE LA PAGE - TO BE DEPRECIATED
@app.post("/delete_chat_history/{chat_id}")
async def delete_chat_history_route(chat_id: str):
    try:
        await delete_all_items_and_adding_first_message(chat_id)
        return {"message": "Chat history deleted successfully"}
    except Exception as e:
        logging.error(f"Erreur lors de la suppression de l'historique du chat : {str(e)}")
        raise HTTPException(status_code=500, detail="Erreur lors de la suppression de l'historique du chat")

# Inclure le routeur de chat
#app.include_router(chat_router)

def create_app():
    return app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8004)
