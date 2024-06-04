from typing import List, Dict, Optional, Tuple, Set
import re
import json

from analytics.tree_structures import Node
from analytics.text_embeddings import get_text_embedding



def create_node(
        index: int, text: str, children_indices: Optional[Set[int]] = None
    ) -> Tuple[int, Node]:
        """Creates a new node with the given index, text, and (optionally) children indices.

        Args:
            index (int): The index of the new node.
            text (str): The text associated with the new node.
            children_indices (Optional[Set[int]]): A set of indices representing the children of the new node.
                If not provided, an empty set will be used.

        Returns:
            Tuple[int, Node]: A tuple containing the index and the newly created node.
        """
        if children_indices is None:
            children_indices = set()

        embeddings = {
            "OpenAI": get_text_embedding(text)
        }
        return (index, Node(text, index, children_indices, embeddings))


def get_text(node_list: List[Node]) -> str:
    """
    Generates a single text string by concatenating the text from a list of nodes.

    Args:
        node_list (List[Node]): List of nodes.

    Returns:
        str: Concatenated text.
    """
    text = ""
    for node in node_list:
        text += f"{' '.join(node.text.splitlines())}"
        text += "\n"
    return text


def get_node_list(node_dict: Dict[int, Node]) -> List[Node]:
    """
    Converts a dictionary of node indices to a sorted list of nodes.

    Args:
        node_dict (Dict[int, Node]): Dictionary of node indices to nodes.

    Returns:
        List[Node]: Sorted list of nodes.
    """
    indices = sorted(node_dict.keys())
    node_list = [node_dict[index] for index in indices]
    return node_list


def get_string_after_colon(input_string):
    # Find the index of the colon
    colon_index = input_string.find(':')

    # Check if the colon is present and extract the substring after it
    if colon_index != -1:
        result_string = input_string[colon_index + 1:].strip()
        return result_string
    return input_string

def extract_plot_data(html_string):
    # Define regular expression pattern to match constants
    pattern = re.compile(r'const\s+(point|hover|label)DataBase64\s*=\s*"([^"]+)"')

    # Find all matches in the HTML string
    matches = pattern.findall(html_string)

    # Initialize dictionary to store extracted data
    extracted_data = {
        "pointDataBase64": None,
        "hoverDataBase64": None,
        "labelDataBase64": None
    }

    # Iterate over matches and populate extracted_data dictionary
    for match in matches:
        data_type, data_value = match
        extracted_data[data_type + "DataBase64"] = data_value

    return extracted_data