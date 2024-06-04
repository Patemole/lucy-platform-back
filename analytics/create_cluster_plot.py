import numpy as np
import logging
import os
import webbrowser
from datetime import datetime, timedelta

from analytics.historic_chat_retrival import get_course_chat_messages
from analytics.clustering import global_cluster_embeddings
from analytics.construct_tree import construct_tree
from analytics.utils import create_node, extract_plot_data
import datamapplot

# Configurer logging pour afficher les messages de journalisation
logging.basicConfig(level=logging.INFO)

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
    # Récupérer les messages du cours
    logging.info(f"Getting Course {course_id} messages")
    messages = get_course_chat_messages(course_id=course_id, start_date=datetime.now() - timedelta(days=200), end_date=datetime.now())
    messages = [message['body'] for message in messages]
    logging.info(f"Received {len(messages)} messages for course: {course_id}")
    if len(messages) == 0:
        return {"error": "No messages for course id " + course_id}

    # Obtenir les étiquettes
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
        labels[:, 1],
        labels[:, 2],
        hover_text=labels[:, 0],
        font_family="UI Sans Serif",
        enable_search=True,
    ))

    #return {'data': extract_plot_data(plot_html)}
    return {'plot_html': plot_html, 'plot_data': extract_plot_data(plot_html)}


if __name__ == "__main__":
    result = create_cluster_plot("course_id_test")
    
    if 'error' in result:
        print(result['error'])
    else:
        output_file = "cluster_plot.html"
        with open(output_file, "w") as file:
            file.write(result['plot_html'])
        
        webbrowser.open('file://' + os.path.realpath(output_file))
