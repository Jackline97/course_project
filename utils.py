from sentence_transformers import util
import torch
from nltk.metrics import edit_distance
from difflib import SequenceMatcher
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import wordnet as wn
import re
from nltk.stem import WordNetLemmatizer
import nltk
import json
import os
from nltk.tokenize import sent_tokenize

# Download necessary NLTK data
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')


lemmatizer = WordNetLemmatizer()

def split_documents_into_chunks(documents, chunk_size=2000, overlap_size=400):
    """
    Splits documents into chunks of approximately chunk_size characters,
    ensuring each chunk ends at a sentence boundary with an overlap of overlap_size characters.
    
    :param documents: List of documents (strings) to be split.
    :param chunk_size: Approximate maximum size of each chunk in characters.
    :param overlap_size: Desired overlap between chunks in characters.
    :return: List of chunks.
    """
    chunks = []
    for document in documents:
        sentences = sent_tokenize(document)
        current_chunk = []
        current_length = 0
        i = 0
        while i < len(sentences):
            sentence = sentences[i]
            sentence_length = len(sentence) + 1  # +1 for space or punctuation
            if current_length + sentence_length > chunk_size:
                # Add the current chunk to chunks
                chunks.append(' '.join(current_chunk).strip())
                # Start new chunk with overlap
                overlap_sentences = []
                overlap_length = 0
                # Collect sentences for overlap from the end of current_chunk
                j = len(current_chunk) - 1
                while j >= 0 and overlap_length < overlap_size:
                    overlap_sentences.insert(0, current_chunk[j])
                    overlap_length += len(current_chunk[j]) + 1
                    j -= 1
                current_chunk = overlap_sentences
                current_length = overlap_length
            else:
                current_chunk.append(sentence)
                current_length += sentence_length
                i += 1
        if current_chunk:
            chunks.append(' '.join(current_chunk).strip())
    return chunks


def check_similarity(sentence_list, target_sentence, similarity_model, threshold=0.8):
    """
    Checks if the target_sentence is dissimilar to all sentences in sentence_list based on the threshold.
    Returns True if dissimilar (i.e., similarity below threshold), False otherwise.
    """
    # Get embeddings
    target_embedding = similarity_model.get_embedding(target_sentence)
    list_embeddings = [similarity_model.get_embedding(s) for s in sentence_list]

    # Stack embeddings for batch processing
    list_embeddings_tensor = torch.stack(list_embeddings)

    # Compute cosine similarities in batch
    cosine_similarities = util.cos_sim(list_embeddings_tensor, target_embedding)
    
    # Check if any similarity exceeds the threshold
    if torch.any(cosine_similarities > threshold):
        return False  # Similar sentence found
    return True  # All sentences are dissimilar

def description_similarity(desc1, desc2):
    """
    Computes cosine similarity between two descriptions using TF-IDF vectorization.

    :param desc1: First description.
    :param desc2: Second description.
    :return: Cosine similarity score.
    """
    vectorizer = TfidfVectorizer(stop_words='english')
    vectors = vectorizer.fit_transform([desc1, desc2])
    cosine_sim = cosine_similarity(vectors[0:1], vectors[1:2])
    return cosine_sim[0][0]


def get_abbreviation(phrase):
    """
    Generates an abbreviation by taking the first letter of each word in the phrase.
    Handles hyphens, slashes, and lemmatizes the abbreviation to handle plurals.

    :param phrase: The input string containing multiple words.
    :return: A string representing the abbreviation.
    """
    words = re.split(r'[\s\-/]+', phrase)
    abbreviation = ''.join(word[0].upper() for word in words if word)
    # Lemmatize the abbreviation to handle plural forms
    abbreviation_lemma = lemmatizer.lemmatize(abbreviation.lower())
    return abbreviation_lemma.upper()

def normalize_name(name):
    """
    Normalizes the entity name by converting to lowercase, removing non-alphanumeric characters,
    and lemmatizing to handle singular and plural forms.

    :param name: The entity name.
    :return: Normalized name.
    """
    # Remove non-alphanumeric characters
    name_clean = re.sub(r'\W+', ' ', name).lower()
    # Tokenize the name
    tokens = nltk.word_tokenize(name_clean)
    # Lemmatize each token
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    # Reconstruct the name
    normalized_name = ' '.join(lemmatized_tokens)
    return normalized_name


def are_synonyms(word1, word2):
    """
    Checks if two words are synonyms using WordNet.

    :param word1: First word.
    :param word2: Second word.
    :return: True if synonyms, False otherwise.
    """
    synsets1 = wn.synsets(word1.lower())
    synsets2 = wn.synsets(word2.lower())
    if not synsets1 or not synsets2:
        return False
    for synset in synsets1:
        for lemma in synset.lemmas():
            if lemma.name().lower() == word2.lower():
                return True
    for synset in synsets2:
        for lemma in synset.lemmas():
            if lemma.name().lower() == word1.lower():
                return True
    return False

def string_similarity(name1, name2):
    """
    Computes similarity between two strings using multiple metrics.

    :param name1: First name.
    :param name2: Second name.
    :return: Average similarity score.
    """
    name1 = name1.lower()
    name2 = name2.lower()

    # Levenshtein Distance
    lev_distance = edit_distance(name1, name2)
    lev_similarity = 1 - lev_distance / max(len(name1), len(name2))

    # Jaro-Winkler Similarity
    jaro_similarity = SequenceMatcher(None, name1, name2).ratio()

    # Average similarity
    avg_similarity = (lev_similarity + jaro_similarity) / 2

    return avg_similarity


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