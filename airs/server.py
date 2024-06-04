#POUR LE STREAM AVEC LANGRAPH, UTILISER L'ENDPOINT @airs_router.post("/send_message_socratic_langgraph")
#MODIFIER ÉGALEMENT LA FONCTION LUCY TOUT EN BAS DU DOCUMENT QUI EST APPELÉ DANS LA DÉFINITION DE L'ENDPOINT
#STREAM => ETAT => NE FONCTIONNE PAS 


import sys
import os

# Ajouter les répertoires nécessaires au chemin d'accès Python
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'airs')))  # Assurez-vous d'ajouter ce chemin
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'third_party_api_clients', 'openai')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'third_party_api_clients', 'pinecone')))

# Afficher le chemin d'accès Python pour vérification
print("\nPython path:", sys.path, "\n")

# Import des modules nécessaires
from airs.session.cleanup_sessions import cleanup_sessions
from airs.session.agent_session import AgentSession
from airs.agent.rag_agent import RAGAgent
from airs.model.input_query import InputQuery
from back_socratic.third_party_api_clients.pinecone.pinecone_api_client_socratic import PineconeApiClient
from airs.database.dynamo_db.chat import get_chat_history, store_message_async
from airs.database.dynamo_db.new_instance_chat import delete_all_items_and_adding_first_message
# from pinecone_embedding.main_embeddings_treatment import embeddings_treatment  # Assurez-vous que ce module est accessible

import asyncio
import threading
from fastapi import APIRouter, FastAPI, Request, Response, UploadFile, HTTPException, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.responses import FileResponse
from uuid import uuid4
from datetime import datetime
import logging
import dropbox
from typing import Dict, Any
from pydantic import BaseModel
from langchain_openai import OpenAIEmbeddings
import os
from openai import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader
from typing import Any, Dict, List, Tuple
from pydantic import BaseModel
from langchain_openai import OpenAIEmbeddings
import langchain_pinecone
from pinecone import Pinecone, Index
import httpx
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

PINECONE_API_URL = os.getenv('PINECONE_API_URL')


class IndexRequest(BaseModel):
    pinecone_index_name: str


PINECONE_POOL_THREADS = 4
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 50
PINECONE_BATCH_SIZE = 64



# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("airs.log")
    ]
)

app = FastAPI(
    title="AI Retrieval Service",
    version="0.0.0"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

airs_router = APIRouter(prefix='/chat', tags=['chat'])



#TRAITEMENT D'UN MESSAGE ÉLÈVE AVEC LA LOGIQUE LANGRAPH
@airs_router.post("/send_message_socratic_langgraph")
async def chat(request: Request, response: Response, input_query: InputQuery) -> StreamingResponse:
    chat_id = input_query.chat_id
    course_id = input_query.course_id  # Get course_id from input_query
    
    # Récupérez l'historique de chat
    chat_history = await get_chat_history(chat_id)
    print(chat_history)
    
    # Stockez le message de manière asynchrone
    asyncio.ensure_future(store_message_async(chat_id, username=input_query.username, course_id=input_query.course_id, message_body=input_query.message))

    # Créez une réponse en streaming en passant l'historique de chat
    return StreamingResponse(lucy(input_query.message, chat_history, course_id), media_type="text/event-stream")


#RÉCUPÉRATION DE L'HISTORIQUE DE CHAT POUR LANGRAPH
@airs_router.get("/get_chat_history/{chat_id}")
async def get_chat_history_route(chat_id: str):
    return await get_chat_history(chat_id)



#SUPPRIMER L'HISTORIQUE DE CHAT À CHAQUE CHARGEMENT DE LA PAGE - à changer dans la logique
@airs_router.post("/delete_chat_history/{chat_id}")
async def delete_chat_history_route(chat_id: str):
    try:
        await delete_all_items_and_adding_first_message(chat_id)
        return {"message": "Chat history deleted successfully"}
    except Exception as e:
        logging.error(f"Erreur lors de la suppression de l'historique du chat : {str(e)}")
        raise HTTPException(status_code=500, detail="Erreur lors de la suppression de l'historique du chat")
    

##########################################################################
#Endpoint pour la création de l'index pinecone
@app.post("/api/create-pinecone-index")
async def create_pinecone_index(request: IndexRequest):
    if not request.pinecone_index_name:
        raise HTTPException(status_code=400, detail="Pinecone index name is required")

    headers = {
        'Api-Key': PINECONE_API_KEY,
        'Content-Type': 'application/json'
    }

    payload = {
        "name": request.pinecone_index_name,
        "dimension": 3072,
        "metric": "cosine",
        "replicas": 1,
        "pod_type": "p1",
        "shards": 1
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(f"{PINECONE_API_URL}/databases", json=payload, headers=headers)

        if response.status_code == 201:
            return {"message": "Pinecone index created successfully"}
        else:
            raise HTTPException(status_code=response.status_code, detail=response.json())
##########################################################################



@airs_router.get("/")
async def default():
    return {"host": "AI Retrieval Service", "Version": 0.0, "Vector Embeddings Index": PineconeApiClient().index.describe_index_stats().to_dict()}

app.include_router(airs_router)












################################################################################################################
# Fonction clustering analytics
################################################################################################################
import random
import logging
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Tuple, Set
import numpy as np
import tiktoken
import umap.umap_ as umap
from sklearn.mixture import GaussianMixture
from threading import Lock
from concurrent.futures import ThreadPoolExecutor
import boto3
from datetime import datetime, timedelta
from botocore.exceptions import ClientError
from tenacity import retry, stop_after_attempt, wait_random_exponential
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI
import re
import webbrowser
import os
import datamapplot
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

#AWS
AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
AWS_REGION = os.getenv('AWS_REGION')

#OpenAI
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Configure logging to print messages to the console
logging.basicConfig(level=logging.INFO)

# Set a random seed for reproducibility
RANDOM_SEED = 224
random.seed(RANDOM_SEED)

SYSTEM_PROMPT = """You are a helpful tool to summarize user interactions in any domain."""

PROMPT = "The provided interactions are written by students to an assistant. Provide a concise summary of WHAT the student is requesting. Do not make mention of the students. Limit the summary to 10 words or fewer: {context}\nSummary: "

class ClusteringAlgorithm(ABC):
    @abstractmethod
    def perform_clustering(self, embeddings: np.ndarray, **kwargs) -> List[List[int]]:
        pass

class Node:
    def __init__(self, text: str, index: int, children: Set[int], embeddings) -> None:
        self.text = text
        self.index = index
        self.children = children
        self.embeddings = embeddings

class Tree:
    def __init__(self, all_nodes, root_nodes, leaf_nodes, num_layers, layer_to_nodes) -> None:
        self.all_nodes = all_nodes
        self.root_nodes = root_nodes
        self.leaf_nodes = leaf_nodes
        self.num_layers = num_layers
        self.layer_to_nodes = layer_to_nodes

class RAPTOR_Clustering(ClusteringAlgorithm):
    def perform_clustering(
        nodes: List[Node],
        embedding_model_name: str,
        max_length_in_cluster: int = 3500,
        tokenizer=tiktoken.get_encoding("cl100k_base"),
        reduction_dimension: int = 10,
        threshold: float = 0.1,
        verbose: bool = False,
    ) -> List[List[Node]]:
        embeddings = np.array([node.embeddings[embedding_model_name] for node in nodes])
        clusters = perform_clustering(embeddings, dim=reduction_dimension, threshold=threshold)
        print(clusters)
        node_clusters = []
        for label in np.unique(np.concatenate(clusters)):
            indices = [i for i, cluster in enumerate(clusters) if label in cluster]
            cluster_nodes = [nodes[i] for i in indices]
            if len(cluster_nodes) == 1:
                node_clusters.append(cluster_nodes)
                continue
            total_length = sum([len(tokenizer.encode(node.text)) for node in cluster_nodes])
            if (total_length > max_length_in_cluster):
                print(f"reclustering cluster with {len(cluster_nodes)} nodes")
                node_clusters.extend(
                    RAPTOR_Clustering.perform_clustering(
                        cluster_nodes, embedding_model_name, max_length_in_cluster
                    )
                )
            else:
                node_clusters.append(cluster_nodes)
        return node_clusters

class DynamoDBClient:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            print("Creating new instance of DynamoDBClient_greg_test")
            try:
                cls._instance = super(DynamoDBClient, cls).__new__(cls)
                cls.client = boto3.resource(
                    'dynamodb',
                    aws_access_key_id=AWS_ACCESS_KEY_ID,
                    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
                    region_name=AWS_REGION
                )
                print("DynamoDBClient initialized successfully")
            except Exception as e:
                print(f"Failed to initialize DynamoDBClient: {e}")
        else:
            print("Using existing instance of DynamoDBClient")
        
        return cls._instance

class OpenAIApiClient:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(OpenAIApiClient, cls).__new__(cls)
            cls.text_embeddings = OpenAIEmbeddings(
                openai_api_key=OPENAI_API_KEY,
                model="text-embedding-ada-002"
            )
            cls.open_ai_client = OpenAI(api_key=OPENAI_API_KEY)
        return cls._instance

def extract_plot_data(html_string):
    pattern = re.compile(r'const\s+(point|hover|label)DataBase64\s*=\s*"([^"]+)"')
    matches = pattern.findall(html_string)
    extracted_data = {"pointDataBase64": None, "hoverDataBase64": None, "labelDataBase64": None}
    for match in matches:
        data_type, data_value = match
        extracted_data[data_type + "DataBase64"] = data_value
    return extracted_data

def get_text_embedding(text, model="text-embedding-3-small"):   
    client = OpenAIApiClient().open_ai_client
    text = text.replace("\n", " ")
    return client.embeddings.create(input=[text], model=model).data[0].embedding

class BaseSummarizationModel(ABC):
    @abstractmethod
    def summarize(self, context, max_tokens=150):
        pass

class GPT3TurboSummarizationModel(BaseSummarizationModel):
    def __init__(self, model="gpt-3.5-turbo"):
        self.model = model

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def summarize(self, context, max_tokens=500, stop_sequence=None):
        try:
            client = OpenAIApiClient().open_ai_client
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": PROMPT.format(context=context)},
                ],
                max_tokens=max_tokens,
            )
            return get_string_after_colon(response.choices[0].message.content)
        except Exception as e:
            print(e)
            return e

class GPT3SummarizationModel(BaseSummarizationModel):
    def __init__(self, model="text-davinci-003"):
        self.model = model

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def summarize(self, context, max_tokens=500, stop_sequence=None):
        try:
            client = OpenAIApiClient().open_ai_client
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful tool to summarize instructions"},
                    {"role": "user", "content": PROMPT.format(context=context)},
                ],
                max_tokens=max_tokens,
            )
            return get_string_after_colon(response.choices[0].message.content)
        except Exception as e:
            print(e)
            return e

def global_cluster_embeddings(embeddings: np.ndarray, dim: int, n_neighbors: Optional[int] = None, metric: str = "cosine") -> np.ndarray:
    num_points = len(embeddings)
    if n_neighbors is None:
        n_neighbors = max(int((num_points - 1) ** 0.5), 2)  # Ensure n_neighbors is always greater than 1
    else:
        n_neighbors = min(n_neighbors, num_points - 1)  # Ensure n_neighbors is less than the number of points
    if n_neighbors < 2:
        n_neighbors = 2  # UMAP requires at least 2 neighbors

    if num_points < 3:
        # Handle very small datasets differently
        return embeddings  # or apply a different reduction technique

    reduced_embeddings = umap.UMAP(n_neighbors=n_neighbors, n_components=dim, metric=metric).fit_transform(embeddings)
    return reduced_embeddings

def local_cluster_embeddings(embeddings: np.ndarray, dim: int, num_neighbors: int = 10, metric: str = "cosine") -> np.ndarray:
    reduced_embeddings = umap.UMAP(n_neighbors=max(num_neighbors, 2), n_components=dim, metric=metric).fit_transform(embeddings)
    return reduced_embeddings

def get_optimal_clusters(embeddings: np.ndarray, max_clusters: int = 50, random_state: int = RANDOM_SEED) -> int:
    max_clusters = min(max_clusters, len(embeddings))
    n_clusters = np.arange(1, max_clusters)
    bics = []
    for n in n_clusters:
        gm = GaussianMixture(n_components=n, random_state=random_state)
        gm.fit(embeddings)
        bics.append(gm.bic(embeddings))
    optimal_clusters = n_clusters[np.argmin(bics)]
    return optimal_clusters

def GMM_cluster(embeddings: np.ndarray, threshold: float, random_state: int = 0):
    n_clusters = get_optimal_clusters(embeddings)
    gm = GaussianMixture(n_components=n_clusters, random_state=random_state)
    gm.fit(embeddings)
    probs = gm.predict_proba(embeddings)
    labels = [np.where(prob > threshold)[0] for prob in probs]
    return labels, n_clusters

def perform_clustering(embeddings: np.ndarray, dim: int, threshold: float, verbose: bool = False) -> List[np.ndarray]:
    reduced_embeddings_global = global_cluster_embeddings(embeddings, dim)
    global_clusters, n_global_clusters = GMM_cluster(reduced_embeddings_global, threshold)
    print(f"Global Clusters: {n_global_clusters}")
    all_local_clusters = [np.array([]) for _ in range(len(embeddings))]
    total_clusters = 0
    for i in range(n_global_clusters):
        global_cluster_embeddings_ = embeddings[np.array([i in gc for gc in global_clusters])]
        if verbose:
            logging.info(f"Nodes in Global Cluster {i}: {len(global_cluster_embeddings_)}")
        if len(global_cluster_embeddings_) == 0:
            continue
        if len(global_cluster_embeddings_) <= dim + 1:
            local_clusters = [np.array([0]) for _ in global_cluster_embeddings_]
            n_local_clusters = 1
        else:
            reduced_embeddings_local = local_cluster_embeddings(global_cluster_embeddings_, dim)
            local_clusters, n_local_clusters = GMM_cluster(reduced_embeddings_local, threshold)
        if verbose:
            logging.info(f"Local Clusters in Global Cluster {i}: {n_local_clusters}")
        for j in range(n_local_clusters):
            local_cluster_embeddings_ = global_cluster_embeddings_[np.array([j in lc for lc in local_clusters])]
            indices = np.where((embeddings == local_cluster_embeddings_[:, None]).all(-1))[1]
            for idx in indices:
                all_local_clusters[idx] = np.append(all_local_clusters[idx], j + total_clusters)
        total_clusters += n_local_clusters
    print(f"Total Clusters: {total_clusters}")
    return all_local_clusters

def construct_tree(
    leaf_nodes: Dict[int, Node],
    summarizer=GPT3TurboSummarizationModel(),
    num_layers=2,
    reduction_dimension=5,
    cluster_embedding_model="OpenAI",
    summarization_length=500,
    use_multithreading: bool = False,
) -> Dict[int, Node]:
    print("Using Cluster TreeBuilder")
    layer_to_nodes = {0: list(leaf_nodes.values())}
    next_node_index = len(leaf_nodes)
    current_level_nodes = leaf_nodes
    def process_cluster(cluster, new_level_nodes, next_node_index, summarization_length, lock):
        node_texts = get_text(cluster)
        summarized_text = summarizer.summarize(context=node_texts, max_tokens=summarization_length)
        __, new_parent_node = create_node(next_node_index, summarized_text, {node.index for node in cluster})
        with lock:
            new_level_nodes[next_node_index] = new_parent_node
    for layer in range(num_layers):
        new_level_nodes = {}
        print(f"Constructing Layer {layer}")
        node_list_current_layer = get_node_list(current_level_nodes)
        if len(node_list_current_layer) <= reduction_dimension + 1:
            print(current_level_nodes)
            print(node_list_current_layer)
            print(f"Stopping Layer construction: Cannot Create More Layers. Total Layers in tree: {layer}")
            break
        print("Node list current layer: ", [(node.index, node.text) for node in node_list_current_layer])
        clusters = RAPTOR_Clustering.perform_clustering(node_list_current_layer, cluster_embedding_model, reduction_dimension=reduction_dimension)
        print(f"Clusters for layer {layer}: {[[node.index for node in cluster] for cluster in clusters]}")
        lock = Lock()
        if use_multithreading:
            with ThreadPoolExecutor() as executor:
                for cluster in clusters:
                    executor.submit(process_cluster, cluster, new_level_nodes, next_node_index, summarization_length, lock)
                    next_node_index += 1
                executor.shutdown(wait=True)
        else:
            for cluster in clusters:
                process_cluster(cluster, new_level_nodes, next_node_index, summarization_length, lock)
                next_node_index += 1
        layer_to_nodes[layer + 1] = list(new_level_nodes.values())
        current_level_nodes = new_level_nodes
    return layer_to_nodes

def create_node(index: int, text: str, children_indices: Optional[Set[int]] = None) -> Tuple[int, Node]:
    if children_indices is None:
        children_indices = set()
    embeddings = {"OpenAI": get_text_embedding(text)}
    return (index, Node(text, index, children_indices, embeddings))

def get_text(node_list: List[Node]) -> str:
    text = ""
    for node in node_list:
        text += f"{' '.join(node.text.splitlines())}"
        text += "\n"
    return text

def get_node_list(node_dict: Dict[int, Node]) -> List[Node]:
    indices = sorted(node_dict.keys())
    node_list = [node_dict[index] for index in indices]
    return node_list

def get_string_after_colon(input_string):
    colon_index = input_string.find(':')
    if colon_index != -1:
        result_string = input_string[colon_index + 1:].strip()
        return result_string
    return input_string

def get_course_chat_messages(course_id: str, start_date: datetime, end_date: datetime = datetime.now()):
    try:
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
            ScanIndexForward=True
        )
        return response.get('Items', [])
    except ClientError as e:
        error_code = e.response['Error']['Code']
        error_message = e.response['Error']['Message']
        print(f"Error querying chats for course with ID {course_id}: {error_code} - {error_message}")
        return []

def get_node_labels(tree, embedding_model="OpenAI"):
    idx_to_node = {node.index: [node.text] for node in tree[len(tree) - 1]}
    num_layers = len(tree)
    layer = num_layers - 1
    while layer > 0:
        layer_labels = {node.index: [node.text] for node in tree[layer - 1]}
        for node in tree[layer]:
            for children in node.children:
                layer_labels[children].extend(idx_to_node[node.index])
        idx_to_node = layer_labels
        layer -= 1
    return [node.embeddings[embedding_model] for node in tree[0]], np.array(list(idx_to_node.values()))

def create_cluster_plot(course_id: str):
    logging.info(f"Getting Course {course_id} messages")
    messages = get_course_chat_messages(course_id=course_id, start_date=datetime.now() - timedelta(days=200), end_date=datetime.now())
    messages = [message['body'] for message in messages]
    logging.info(f"Received {len(messages)} messages for course: {course_id}")
    if len(messages) == 0:
        return {"error": "No messages for course id " + course_id}
    logging.info(f"Getting Embeddings for course: {course_id}")
    leaf_nodes = {}
    for index, text in enumerate(messages):
        _, node = create_node(index, text)
        leaf_nodes[index] = node
    logging.info(f"Constructing RAPTOR tree for course: {course_id}")
    tree = construct_tree(leaf_nodes)
    logging.info(f"Getting node labels for course: {course_id}")
    embeddings, labels = get_node_labels(tree)
    logging.info(f"Getting Reduced Embeddings for course: {course_id}")
    reduced_embeddings = global_cluster_embeddings(embeddings=embeddings, dim=2)
    logging.info(f"Creating interactive plot for course: {course_id}")
    plot_html = str(datamapplot.create_interactive_plot(
        reduced_embeddings,
        labels[:, 0],  # Changed to access the first column, adjust as needed
        labels[:, 1],  # Changed to access the second column, adjust as needed
        hover_text=labels[:, 0],
        font_family="UI Sans Serif",
        enable_search=True,
    ))
    output_file = "/Users/gregoryhissiger/Socratic-demo-1/generated_html/cluster_plot.html"
    with open(output_file, "w") as file:
        file.write(plot_html)
    logging.info(f"Graph for course {course_id} saved to {output_file}")
    return {'plot_html': plot_html, 'plot_data': extract_plot_data(plot_html)}


table = DynamoDBClient().client.Table("PROD_chat_socratic")
################################################################################################################
# end Fonction clustering analytics
################################################################################################################














################################################################################################################
# Fonction socratic chat
################################################################################################################

import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
import sys
import logging
import langchain_pinecone
from pinecone import Pinecone, Index
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI
import time
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.schema import Document
from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict
from typing import List
from langchain.memory import ChatMessageHistory
import uuid
import asyncio
import nest_asyncio
from aiostream import stream
from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse
import anyio
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

#Groq
GROQ_API_KEY = os.getenv('GROQ_API_KEY')

#Pinecone
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')

#OpenAI
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Set API Key
os.environ["GROQ_API_KEY"] = GROQ_API_KEY


# Initialize Groq Langchain chat object and conversation
model = "llama3-70b-8192"
GROQ_LLM = ChatGroq(model_name=model, streaming=True)
print(GROQ_LLM)

logging.basicConfig(level=logging.INFO)

class PineconeApiClientSocratic:
    _instances = {}

    def __new__(cls, course_id, *args, **kwargs):
        if course_id not in cls._instances:
            logging.info("Initialisation de l'instance PineconeApiClient avec la logique socratic...")
            instance = super(PineconeApiClientSocratic, cls).__new__(cls)
            try:
                pinecone = Pinecone(api_key=PINECONE_API_KEY)

                # Determine the index name based on the course_id
                if course_id == "healthcare":
                    index_name = "ai-in-healthcare"
                elif course_id == "sustainable":
                    index_name = "sustainable-design"
                elif course_id == "business":
                    index_name = "business-design"
                else:
                    index_name = "ai-in-healthcare"

                logging.info(f"Using index: {index_name}")

                instance.index = pinecone.Index(index_name)
                text_embeddings = OpenAIApiClient().text_embeddings
                instance.vectorstore = langchain_pinecone.Pinecone(instance.index, text_embeddings, "text")
                logging.info("PineconeApiClient a été initialisé avec succès.")
                cls._instances[course_id] = instance
            except Exception as e:
                logging.error(f"Erreur lors de l'initialisation de PineconeApiClient: {e}")
                raise e
        return cls._instances[course_id]

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

# Define a decorator to measure the execution time of functions
def timing_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Execution time for: {end_time - start_time:.4f} seconds")
        return result
    return wrapper

# Prompt and chain for finding methodology
methodology_prompt = PromptTemplate(
    template="""system
    You are an experienced knowledge master and knows everything that is teached at an university in the world.
        You are a master at finding good methodology and the solution to all the steps of the methodology 
        to find the solution and the path to answer any problem that is asked to you.
     
    user
    You will follow the steps below to find the following information:\n
    - The problem the student is trying to solve. \n
    - The methodology to solve the problem. \n
    - The necessary information to answer the problem \n
    - The final solution to the problem. \n
    - The guidance to complete the current step of the methodology. \n

    The steps are:
    1. Put together the information from the MEMORY which is the history of your conversation with the student, INFO_RETRIEVED which the retrieved piece of information from the class content and the USER_INPUT which is the student question. \n
    2. Find the problem the student is trying to solve from step 1. \n
    3. Find the necessary information to answer the problem in step 1 from INFO_RETRIEVED. Some informations might be unreleated your job is to find the iforamtion that answer the student question. \n
    3. Find all the information related to the problem from step 2. in the MEMORY and the INFO_RETRIEVED and the USER_INPUT.
    4. Conduct a thorough research to find the methodology to solve the problem from step 2.
    5. Find the answer to all the steps of the methodology found in step 4 get your information needed for your answers based on INFO_RETRIEVED.
    6. Determine at which step of the methodology the USER_INPUT is.
    7. Provide guidance to complete the step found at step 6. at which the user is. and the information necessary to answer the question

    Output a single answer of the following format:
        "problem": "...", which is the problem the student is seeking help for
        "steps": "step_1": "...", "step_2": "...", ...,"final_step": "...", which is the steps to tkae to answer the question and can vary on the question
        "answers": "step_1_answer": "...", "step_2_answer": "...", ..., "final_step_answer": "...", this is the answers step to answer the question the information to answer it need from INFO_RETRIEVED
        "guidance": "..." which is the guidance to guide the student to the answer
    
    Return a JSON with no premable ou explanation of the following above:
    
    user
    USER_INPUT:\n\n {user_input} \n\n
    INFO_RETRIEVED: {info_retrieved} \n\n
    MEMORY:\n\n {memory} \n\n
    assistant
    
    assistant
    """,
    input_variables=["user_input", "memory", "info_retrieved"],
)

methodology_chain = methodology_prompt | GROQ_LLM | JsonOutputParser()

# Decorated function to invoke the input category chain
@timing_decorator
def invoke_chain_methodology(data):
    return methodology_chain.invoke(data)

# Prompt and chain for drafting socratic answer
draft_answer_socratic_prompt = PromptTemplate(
    template="""system
    You are a master in the Socratic method, skilled at guiding students to find answers on their own. However, you also know when to provide direct answers for simple, factual questions.
    Act as a teacher with 20+ years of experience teaching students. 
    You must never tell the answer of a question to the student but rather give him explaination of the concepts from his prolem and 
        you must guide him through the methodology to find the solution on its own.
    You must always engage his reasoning for him to find the answer on his own. 

    user
    First of category thoughfully the question if it is about general class information or general information, such as dates, exam structures, or deadlines this is a DIRECT_QUESTION ou if it is quizzes, assignments, ou exam questions this is EXAM_QUESTION
    If the question is DIRECT_QUESTION, about general information, such as dates, exam structures, ou deadlines, provide a direct, concise answer using the information from the ANSWERS. \n
    For EXAM_QUESTION questions about quizzes, assignments, ou exam content, follow the Socratic method. Use the USER_INPUT, ANSWERS, and MEMORY to guide the student to the answer using METHODOLOGY and GUIDANCE. \n
    Verify that the USER_INPUT correlates with the PROBLEM. If not, find if it correlates with another problem from the ANSWERS, MEMORY, ou USER_INPUT. \n 
    Compare the USER_INPUT and PROBLEM to the ANSWERS. If there's a discrepancy, ask the student to double-check their information. \n
    Do not provide the answers directly for complex questions. Instead, guide the student through questions to help them find the answer. \n
    If any information has already been given in the MEMORY, provide it to the student. \n
    Always engage the student's reasoning for them to find the answer on their own. \n
    Format for DIRECT_QUESTION:\n

    Answer directly and concisely for simple questions.\n\n
    
    Format for EXAM_QUESTION: \n

    Briefly set the context of the MAIN_PROBLEM.
    Summarize all relevant information.
    Provide detail guidance for the student based on the information provided. \n\n
    Go to the line and make nice formated paragraph for each part. \n
    
    Example Output for Simple Questions:

    "The midterm exam is on October 15th, and it will cover chapters 1 through 5. the exam is 1 hlong "\n\n
    
    Example Output for Socratic Method Questions: 

    "You're asking about solving a particular type of problem in calculus. \n
    We know that to solve this problem, we need to apply the chain rule. \n
    Can you identify the functions that need to be differentiated in this problem?" \n
    user
    USER_INPUT: {user_input} \n\n
    PROBLEM: {problem} \n
    METHODOLOGY: {methodology} \n
    ANSWERS: {answers} \n
    GUIDANCE: {guidance} \n\n
    MEMORY: {memory} \n\n\
    assistant""",
    input_variables=["user_input", "memory", "problem", "methodology", "answers", "guidance"],
)

draft_answer_socratic_chain = draft_answer_socratic_prompt | GROQ_LLM | StrOutputParser()

def invoke_chain_draft_answer_socratic(data):
    return draft_answer_socratic_chain.invoke(data)

def query_pinecone(query, course_id):
    client = OpenAIApiClient()
    query_embedding = client.text_embeddings.embed_query(query)
    index = PineconeApiClientSocratic(course_id).index
    retrieved_docs = PineconeApiClientSocratic(course_id).vectorstore.similarity_search(query=query, k=2)
    return retrieved_docs

def rag_info(data, course_id):
    responses = query_pinecone(data, course_id)
    return responses
    
rag_info_retrieval_prompt = PromptTemplate(
    template="""system
    You are a filtering step in a llm chain. You are analyzing text to retrieve the most accurate and useful information.
    You are a master in finding the information that you need to answer a student question.
    You have 20+ years of experience of analysing text and outlining the best information given a question. 
    
    user
    Given the USER_INPUT and the RAG_INFO outline the information from the RAG_INFO that is needed to answer the question in USER_INPUT. 
    Identify in a very detailed manner what information in the text is relevant to the question, there will be useless informations, do not consider them.
    If something in the Information is not relevant do not take it into account. Some informaiton in the RAG_INFO might not be realted you have to find what is related to the USER_INPUT.
    Be consice and do not extend in explanation just state all information needed without adding more wihtout any other detail no need to introduce ou conclude your answer. \n
    user
    USER_INPUT: {user_input} \n\n
    RAG_INFO: {rag_info} \n\n
    \n\n
    assistant""",
    input_variables=["user_input", "rag_info"],
)

rag_info_chain = rag_info_retrieval_prompt | GROQ_LLM | StrOutputParser()

def invoke_chain_rag_info(data):
    return rag_info_chain.invoke(data)

class GraphState(TypedDict):
    user_input: str
    guidance: str
    problem: str
    course_id: str
    info_retrieved: str
    methodology: str
    socratic_answer_given: str
    answers: str
    memory: List[str]

def state_printer(state):
    print("---STATE PRINTER---")
    print(f"user_input: {state['user_input']} \n")
    print(f"memory: {state['memory']} \n")
    print(f"problem: {state['problem']} \n")
    print(f"methodology: {state['methodology']} \n")
    print(f"guidance: {state['guidance']} \n")
    print(f"answers: {state['answers']} \n")
    print(f"socratic_answer_given: {state['socratic_answer_given']} \n")
    print(f"info_retrieved: {state['info_retrieved']} \n")
    return

history = ChatMessageHistory()

def user_input_question(state):
    print("---USER INPUT---")
    user_input = state["user_input"]
    course_id = state["course_id"]
    history.add_user_message(user_input)
    memory = history.messages
    print(user_input)
    return {"user_input": user_input, "memory": memory, "course_id": course_id}

def methodology_and_guidance(state):
    user_input = state["user_input"]
    info_retrieved = state["info_retrieved"]
    memory = state["memory"]
    methodology_and_guidance = invoke_chain_methodology({"user_input": user_input, "memory": memory, "info_retrieved": info_retrieved})
    problem = methodology_and_guidance["problem"]
    methodology = methodology_and_guidance["steps"]
    answers = methodology_and_guidance["answers"]
    guidance = methodology_and_guidance["guidance"]
    return {"methodology": methodology, "guidance": guidance}

def rag_info_retriever(state):
    user_input = state["user_input"]
    course_id = state["course_id"]
    rag_info = query_pinecone(user_input, course_id)
    info = invoke_chain_rag_info({"user_input": user_input, "rag_info": rag_info})
    print(info)
    return {"info_retrieved": info}

def draft_answer_socratic(state):
    user_input = state["user_input"]
    memory = state["memory"]
    problem = state["problem"]
    methodology = state["methodology"]
    answers = state["answers"]
    guidance = state["guidance"]
    draft_answer_socratic = invoke_chain_draft_answer_socratic({"user_input": user_input, "memory": memory, "problem": problem, "methodology": methodology, "answers": answers, "guidance": guidance})
    history.add_ai_message(draft_answer_socratic)
    memory = history.messages
    return {"socratic_answer_given": draft_answer_socratic, "memory": memory}

workflow = StateGraph(GraphState)

# Define the nodes
workflow.add_node("user_input_question", user_input_question)
workflow.add_node("methodology_and_guidance", methodology_and_guidance)
workflow.add_node("rag_info_retriever", rag_info_retriever)
workflow.add_node("draft_answer_socratic", draft_answer_socratic)

# Add Edges
workflow.set_entry_point("user_input_question")
workflow.add_edge("user_input_question", "rag_info_retriever")
workflow.add_edge("rag_info_retriever", "methodology_and_guidance")
workflow.add_edge("methodology_and_guidance", "draft_answer_socratic")
workflow.add_edge("draft_answer_socratic", END)

# Compile the workflow
app_socratic = workflow.compile()


#DERNIERE FONCTION QUI MARCHE 23 MAI 17H56
@timing_decorator
async def lucy(user_input, chat_history, course_id):
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    inputs = {"user_input": user_input, "chat_history": chat_history, "course_id": course_id}
    full_response = ""

    try:
        async for output in app_socratic.astream_log(inputs, config=config, with_streamed_output_list=True, include_types=["llm"]):
            print("1 - output\n")
            print(output)

            for op in output.ops:
                if op["path"].startswith("/logs/") and op["path"].endswith("/streamed_output/-"):
                    # because we chose to only include LLMs, these are LLM tokens
                    chunk = op["value"].content
                    full_response += chunk
                    print(f"\033[94mLog Streamed Output: {chunk}\033[0m")
                    yield chunk  # Envoie le chunk directement sans préfixe ou suffixe

        # Ensure at least one yield happens
        print(f"\033[92mFull Response: {full_response}\033[0m")
        yield ""  # Envoie une chaîne vide pour indiquer la fin du stream
    except Exception as e:
        print(f"\033[91mException occurred: {e}\033[0m")
        yield f"Error occurred: {e}\n\n"

################################################################################################################
# Fin Fonction socratic chat
################################################################################################################












