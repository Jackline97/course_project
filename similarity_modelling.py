from sentence_transformers import SentenceTransformer, util
from utils import normalize_name, get_abbreviation, are_synonyms, string_similarity, description_similarity


class NodeSimilarity:
    def __init__(self):
        pass  # No LLM initialization required

    def advanced_similarity(self, node1, node2, threshold=0.85):
        entity_name1, entity_type1, entity_description1 = node1
        entity_name2, entity_type2, entity_description2 = node2
        

        # Normalize entity names
        name1_normalized = normalize_name(entity_name1)
        name2_normalized = normalize_name(entity_name2)
        

        # 1. Abbreviation Detection
        abbreviation1 = get_abbreviation(name1_normalized).lower().strip()
        abbreviation2 = get_abbreviation(name2_normalized).lower().strip()
        
        if name1_normalized == abbreviation2 or name1_normalized == abbreviation2+'s':
            return 1
        if name2_normalized == abbreviation1 or name2_normalized == abbreviation1+'s':
            return 1

        # 2. Synonym Detection
        if are_synonyms(name1_normalized, name2_normalized):
            return 1

        # 3. String Similarity of Names
        name_sim = string_similarity(name1_normalized, name2_normalized)
        if name_sim >= threshold:
            return 1

        # 4. Description Similarity
        desc_sim = description_similarity(entity_description1, entity_description2)
        if desc_sim >= threshold:
            return 1
        # 5. If none of the above, they are considered not similar
        return 0


class BertSimilarity:
    """
    A class to compute and cache sentence embeddings using SentenceTransformer.
    """
    def __init__(self, model_name='paraphrase-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.embeddings_cache = {}

    def get_embedding(self, sentence):
        """
        Retrieves the embedding for a sentence, computing and caching it if necessary.
        """
        if sentence not in self.embeddings_cache:
            self.embeddings_cache[sentence] = self.model.encode(sentence, convert_to_tensor=True)
        return self.embeddings_cache[sentence]

    def basic_similarity(self, sentence1, sentence2):
        """
        Computes the cosine similarity between two sentences.
        """
        if sentence1 in sentence2 or sentence2 in sentence1:
            return 1
        
        embedding1 = self.get_embedding(sentence1)
        embedding2 = self.get_embedding(sentence2)
        cosine_sim = util.cos_sim(embedding1, embedding2)
        return cosine_sim.item()
    

    