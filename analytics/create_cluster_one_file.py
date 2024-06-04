

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

class OpenAIApiClient:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(OpenAIApiClient, cls).__new__(cls)
            cls.text_embeddings = OpenAIEmbeddings(
                openai_api_key="sk-proj-EbsswciW1QbyS50aDohkT3BlbkFJgawGIcLqmGMMNjCEe00M",
                model="text-embedding-ada-002"
            )
            cls.open_ai_client = OpenAI(api_key="sk-proj-EbsswciW1QbyS50aDohkT3BlbkFJgawGIcLqmGMMNjCEe00M")
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

if __name__ == "__main__":
    result = create_cluster_plot("course_id_test")
    if 'error' in result:
        print(result['error'])
    else:
        output_file = "cluster_plot.html"
        with open(output_file, "w") as file:
            file.write(result['plot_html'])
        webbrowser.open('file://' + os.path.realpath(output_file))
