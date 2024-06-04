from threading import Lock
from concurrent.futures import ThreadPoolExecutor
import copy
from typing import Dict

from analytics.clustering import RAPTOR_Clustering
from analytics.summarization import GPT3TurboSummarizationModel
from analytics.tree_structures import Node
from analytics.utils import create_node, get_text
from analytics.utils import get_node_list

def construct_tree(
        leaf_nodes: Dict[int, Node],
        summarizer = GPT3TurboSummarizationModel(),
        num_layers = 2,
        reduction_dimension = 5,
        cluster_embedding_model = "OpenAI",
        summarization_length = 500,
        use_multithreading: bool = False,
    ) -> Dict[int, Node]:
        print("Using Cluster TreeBuilder")

        layer_to_nodes = {0 : list(leaf_nodes.values())}
        next_node_index = len(leaf_nodes)
        current_level_nodes = leaf_nodes

        def process_cluster(
            cluster, new_level_nodes, next_node_index, summarization_length, lock
        ):
            node_texts = get_text(cluster)

            summarized_text = summarizer.summarize(
                context=node_texts,
                max_tokens=summarization_length,
            )

            __, new_parent_node = create_node(
                next_node_index, summarized_text, {node.index for node in cluster}
            )

            with lock:
                new_level_nodes[next_node_index] = new_parent_node

        for layer in range(num_layers):

            new_level_nodes = {}

            print(f"Constructing Layer {layer}")

            node_list_current_layer = get_node_list(current_level_nodes)

            if len(node_list_current_layer) <= reduction_dimension + 1:
                print(current_level_nodes)
                print(node_list_current_layer)
                print(
                    f"Stopping Layer construction: Cannot Create More Layers. Total Layers in tree: {layer}"
                )
                break
            print("Node list current layer: ", [(node.index, node.text) for node in node_list_current_layer])
            clusters =  RAPTOR_Clustering.perform_clustering(
                node_list_current_layer,
                cluster_embedding_model,
                reduction_dimension=reduction_dimension
            )

            print(f"Clusters for layer {layer}: {[[node.index  for node in cluster ]for cluster in clusters]}")

            lock = Lock()

            if use_multithreading:
                with ThreadPoolExecutor() as executor:
                    for cluster in clusters:
                        executor.submit(
                            process_cluster,
                            cluster,
                            new_level_nodes,
                            next_node_index,
                            summarization_length,
                            lock,
                        )
                        next_node_index += 1
                    executor.shutdown(wait=True)

            else:
                for cluster in clusters:
                    process_cluster(
                        cluster,
                        new_level_nodes,
                        next_node_index,
                        summarization_length,
                        lock,
                    )
                    next_node_index += 1
            layer_to_nodes[layer + 1] = list(new_level_nodes.values())
            current_level_nodes = new_level_nodes


        return layer_to_nodes
        