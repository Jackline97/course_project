from nltk.tokenize import sent_tokenize
import os
import re

def load_prompt(pth):
    with open(pth, 'r', encoding='utf-8') as file:
        content = file.read()
        return content    

def process_entity_extraction_prompt(content):
    DEFAULT_TUPLE_DELIMITER = "<|>"
    DEFAULT_RECORD_DELIMITER = "##"
    DEFAULT_COMPLETION_DELIMITER = "<|COMPLETE|>"
    content = content.replace('{completion_delimiter}', DEFAULT_COMPLETION_DELIMITER)
    content = content.replace('{tuple_delimiter}', DEFAULT_TUPLE_DELIMITER)
    content = content.replace('{record_delimiter}', DEFAULT_RECORD_DELIMITER)
    return content
    
    
def load_document(folder_path):
    # Initialize a list to store the contents of each text file
    text_files_content = []
    # Loop through each file in the directory
    for filename in os.listdir(folder_path):
        # Check if the file is a text file
        if filename.endswith('.txt'):
            # Construct full file path
            file_path = os.path.join(folder_path, filename)
            # Open the file and read its contents
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                text_files_content.append(content)
    return text_files_content


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


def entity_type_extraction(LLM, prompt, document, domain_name):
    prompt = prompt.replace('{domain}', domain_name)
    prompt = prompt.replace('{document}', document)

    messages=[
        {"role": "user", "content": prompt},
    ]    
    response = LLM(messages, max_new_tokens=100, 
                   pad_token_id=LLM.tokenizer.eos_token_id)
    entities_types = response[0]['generated_text'][-1]['content']
    return entities_types.split('\n\n')[1]