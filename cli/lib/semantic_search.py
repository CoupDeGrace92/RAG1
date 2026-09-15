from sentence_transformers import SentenceTransformer
from pathlib import Path
import numpy as np
import os
from utils.search_utils import load_movies

class SemanticSearch:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.embeddings = None
        self.documents = None
        self.document_map = {}

    def generate_embedding(self, text: str):
        embedding = self.model.encode([text])
        return embedding[0]

    def build_embeddings(self, documents):
        self.documents = documents
        doc_list = []
        for doc in documents:
            id = doc.get("id")
            if id:
                self.document_map[id] = doc

                doc_string = f"{doc.get('title')}: {doc.get('description')}"
                doc_list.append(doc_string)

        self.embeddings = self.model.encode(doc_list, show_progress_bar=True)
        cache_dir = Path("./cache")
        fp = cache_dir/"movie_embeddings.npy"
        
        np.save(fp, self.embeddings)

    def load_or_create_embeddings(self, documents):
        self.documents = documents
        doc_list = []
        warned = False
        for doc in documents:
            id = doc.get("id")
            if id:
                self.document_map[id] = doc

                doc_string = f"{doc.get('title')}: {doc.get('description')}"
                doc.list.append(doc_string)
            else:
                if not warned:
                    print("id missing from a document(s) in ingested data")
                    warned = True
        cache_path = "./cache/movie_embeddings.npy"
        if os.path.exists(cache_path):
            self.embeddings = np.load(cache_path)
            if len(self.embeddings) != len(self.documents):
                print(f"length of cached embeddings ({len(self.embeddings)}) different than document list ({len(self.documents)})")
                self.embeddings = self.model.encode(doc_list, show_progress_bar=True)
        else:
            self.embeddings = self.model.encode(doc_list, show_progress_bar=True)

        return self.embeddings
        
        

def verify_model():
    s_model = SemanticSearch()

    print(f"Model loaded: {s_model.model}")
    print(f"Max sequence length: {s_model.model.max_seq_length}")

def embed_text(text: str) -> None:
    s_model = SemanticSearch()
    embedding = s_model.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")

def verify_embeddings():
    s_model = SemanticSearch()
    movie_fp = "./data/movies.json"
    documents = load_movies(movie_fp)
    embeddings = s_model.load_or_create_embeddings(documents)
    print(f"Number of docs:   {len(documents)}")
    print(
        f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions"
    )