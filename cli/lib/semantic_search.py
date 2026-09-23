from sentence_transformers import SentenceTransformer
from pathlib import Path
import numpy as np
import os,json
from utils.search_utils import load_movies, semantic_chunk, format_search_result
from constants import SCORE_PRECISION

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
                doc_list.append(doc_string)
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
            np.save(cache_path, self.embeddings)

        return self.embeddings

    def search(self, query, limit):
        if self.embeddings is None:
            raise ValueError("No embeddings loaded.  Call 'load_or_create_embeddings' first.")
        if self.documents == None:
            raise ValueError("No Documents loaded")
        
        q_embedding = self.generate_embedding(query)

        scores = []

        for i in range(len(self.embeddings)):
            similarity_score = cosine_similarity(q_embedding, self.embeddings[i])
            s_tuple = (similarity_score, self.documents[i])
            scores.append(s_tuple)

        scores.sort(key=lambda t: t[0], reverse=True)
        scores = scores[:limit]
        out = []
        for score in scores:
            out.append({"score": score[0], "title": score[1].get("title"), "description": score[1].get("description")})

        return out

class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        super().__init__()
        self.model = SentenceTransformer(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata = None

    def build_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents
        chunk_list = []
        chunk_meta = []
        for doc in documents:
            id = doc.get("id")
            if id:
                self.document_map[id] = doc
                desc = doc.get("description")
                if isinstance(desc, str):
                    chunks = semantic_chunk(desc, 4, 1)
                    chunk_list.extend(chunks)
                    id = doc.get("id")
                    movie_idx = self.documents.index(doc)
                    for chunk in chunks:
                        dict = {"movie_idx": movie_idx, "chunk_idx": chunks.index(chunk), "total_chunks": len(chunks)}
                        chunk_meta.append(dict)

        self.chunk_embeddings = self.model.encode(chunk_list, show_progress_bar=True)
        self.chunk_metadata = chunk_meta

        np.save("./cache/chunk_embeddings.npy", self.chunk_embeddings)
        with open("./cache/chunk_metadata.json", "w") as file:
            json.dump({"chunks": chunk_meta, "total_chunks": len(chunk_list)}, file, indent=2)
        return self.chunk_embeddings

    def load_or_create_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents
        chunk_list = []
        chunk_meta = []
        for doc in documents:
            id = doc.get("id")
            if id:
                self.document_map[id] = doc
                desc = doc.get("description")
                if isinstance(desc, str):
                    chunks = semantic_chunk(desc, 4, 1)
                    chunk_list.extend(chunks)
                    id = doc.get("id")
                    movie_idx = self.documents.index(doc)
                    for chunk in chunks:
                        dict = {"movie_idx": movie_idx, "chunk_idx": chunks.index(chunk), "total_chunks": len(chunks)}
                        chunk_meta.append(dict)

        chunk_fp = "./cache/chunk_embeddings.npy"
        meta_fp = "./cache/chunk_metadata.json"
        if os.path.exists(chunk_fp):
            self.chunk_embeddings = np.load(chunk_fp)
        else:
            self.chunk_embeddings = self.model.encode(chunk_list, show_progress_bar=True)
            np.save(chunk_fp, self.chunk_embeddings)
        if os.path.exists(meta_fp):
            with open(meta_fp, "r") as file:
                try:
                    loaded = json.load(file)
                    self.chunk_metadata = loaded.get("chunks")
                except Exception as e:
                    print(f"Error converting {meta_fp} to json: {type(e).__name__} - {e}")
        else:
            self.chunk_metadata = chunk_meta
            with open(meta_fp, "w") as file:
                json.dump({"chunks": chunk_meta, "total_chunks": len(chunk_list)}, file, indent=2)

        return self.chunk_embeddings

    def search_chunks(self, query: str, limit: int=10):
        q_embedding = self.generate_embedding(query)
        chunk_scores = []
        if (self.chunk_embeddings is None) or (self.chunk_metadata is None):
            raise ValueError("No chunk embeddings to search")

        for idx, embedding in enumerate(self.chunk_embeddings):
            c_sim = cosine_similarity(embedding, q_embedding)
            chunk_scores.append({"chunk_idx": idx, "movie_idx": self.chunk_metadata[idx].get("movie_idx"), "score": c_sim})
        chunk_map = {}
        for score in chunk_scores:
            if (chunk_map.get(score.get("movie_idx")) is None) or (chunk_map.get(score.get("movie_idx")) < score.get("score")):
                chunk_map[score.get("movie_idx")] = score.get("score")
        sort_scores = sorted(chunk_map.items(), key=lambda item: item[1], reverse=True)
        if limit < len(sort_scores):
            sort_scores = sort_scores[:limit]
        out = []
        for item in sort_scores:
            out.append(format_search_result(
                self.documents[item[0]].get("id"),
                self.documents[item[0]].get("title"),
                self.documents[item[0]].get("description")[:100],
                item[1]
            ))
        return out

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

def embed_text_query(query):
    s_model = SemanticSearch()
    embedding = s_model.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Shape: {embedding.shape}")

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)