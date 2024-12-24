import os
from pathlib import Path
import json
from similarity_modelling import BertSimilarity, NodeSimilarity
from load_file import load_document, load_prompt, process_entity_extraction_prompt
from extract_entity_relationship import extract_elements_from_chunks, collect_elements_relationship
from utils import split_documents_into_chunks
from prompt_tuning import entity_type_extraction
from build_graph import graph_init_merging, graph_deep_merging

CHECKPOINT_PATH = Path("checkpoint/meta_info.json")
EXTRACTION_PROMPT_PATH = Path('entity_extraction_prompt.txt')
TYPE_EXTRACTION_PROMPT_PATH = Path('entity_type_extraction_prompt.txt')

def initialize_similarity_models():
    """Initialize similarity models."""
    bert_similarity_model = BertSimilarity()
    syntatic_sim_model = NodeSimilarity()
    return bert_similarity_model, syntatic_sim_model

def load_prompts():
    """Load and process extraction prompts."""
    extraction_prompt = load_prompt(EXTRACTION_PROMPT_PATH)
    processed_extraction_prompt = process_entity_extraction_prompt(extraction_prompt)
    type_extraction_prompt = load_prompt(TYPE_EXTRACTION_PROMPT_PATH)
    return processed_extraction_prompt, type_extraction_prompt

def load_and_process_documents(folder_path, chunk_size=2000, overlap_size=400):
    """Load documents and split them into sentence-aware chunks."""
    documents = load_document(folder_path)
    chunks = split_documents_into_chunks(documents, chunk_size=chunk_size, overlap_size=overlap_size)
    return documents, chunks


def save_meta_info(meta_info, file_path='meta_info.json'):
    """
    Save the meta_info dictionary to a JSON file.

    Args:
        meta_info (dict): The meta information to save.
        file_path (str): The path to the JSON file where data will be saved.
    """
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Open the file in write mode with UTF-8 encoding
        with open(file_path, 'w', encoding='utf-8') as f:
            # Serialize and write the dictionary to the file with indentation for readability
            json.dump(meta_info, f, ensure_ascii=False, indent=4)
        
        print(f"meta_info successfully saved to {file_path}")
    
    except TypeError as te:
        print("Serialization Error: Ensure all data in meta_info is JSON-serializable.")
        print(te)
    
    except Exception as e:
        print("An unexpected error occurred while saving meta_info:")
        print(e)
        
    
def load_meta_info(file_path='meta_info.json'):
    """
    Load the meta_info dictionary from a JSON file.

    Args:
        file_path (str): The path to the JSON file to load.

    Returns:
        dict: The loaded meta_info dictionary.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            meta_info = json.load(f)
        print(f"meta_info successfully loaded from {file_path}")
        return meta_info
    
    except FileNotFoundError:
        print(f"The file {file_path} does not exist.")
    
    except json.JSONDecodeError as jde:
        print("Error decoding JSON. Ensure the file is properly formatted.")
        print(jde)
    
    except Exception as e:
        print("An unexpected error occurred while loading meta_info:")
        print(e)


def extract_meta_info(LLM, folder_path, domain_name='Artificial Intelligence', chunk_size=2000, overlap_size=400, max_gleanings=3, sim_threshold = 0.85):
    """Extract meta information from documents."""
    bert_similarity_model, syntatic_sim_model = initialize_similarity_models()
    extraction_prompt, type_extraction_prompt = load_prompts()
    documents, chunks = load_and_process_documents(folder_path, chunk_size=chunk_size, overlap_size=overlap_size)
    
    # Extract entity types
    entity_type = entity_type_extraction(
        LLM,
        type_extraction_prompt,
        document='\n'.join(documents),
        domain_name=domain_name
    )
    
    # Update extraction prompt with entity types
    extraction_prompt = extraction_prompt.replace('{entity_types}', entity_type)
    
    # Extract raw entity-relationship elements from chunks
    elements = extract_elements_from_chunks(LLM, chunks, extraction_prompt, max_gleanings=max_gleanings)
    
    # Collect graph info (node and edges) from raw info (text)
    groups = collect_elements_relationship(elements)
    
    # Merge graph information by removing duplicate nodes and edges using a BERT-based sentence similarity model to evaluate corresponding node and edge descriptions-ONLY.
    final_node, final_edge = graph_init_merging(groups, bert_similarity_model, threshold = sim_threshold)
    
    # Clean the graph (merge nodes based on node name and type )
    merged_nodes, merged_edges = graph_deep_merging(final_node, final_edge, syntatic_sim_model, threshold = sim_threshold)
    
    # Compile meta information
    meta_info = {
        "entity_type": entity_type,       # string
        "elements": elements,             # list
        "groups": groups,                 # dict
        "final_nodes": final_node,        # dict
        "final_edges": final_edge,        # dict
        "merged_nodes": merged_nodes,      # dict
        "merged_edges": merged_edges       # dict
    }
    
    return meta_info

def main(folder_path, LLM):
    """Main function to load or generate meta information."""
    try:
        if CHECKPOINT_PATH.exists():
            meta_info = load_meta_info(file_path=str(CHECKPOINT_PATH))
            print("Loaded meta information from checkpoint.")
        else:
            meta_info = extract_meta_info(LLM, folder_path)
            save_meta_info(meta_info, file_path=str(CHECKPOINT_PATH))
            print("Extracted and saved meta information.")
        return meta_info
    except Exception as e:
        print(f"An error occurred: {e}")
        # Optionally, handle specific exceptions or re-raise
        raise