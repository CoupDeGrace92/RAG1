from sentence_transformers import SentenceTransformer

class SemanticSearch:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

def verify_model():
    s_model = SemanticSearch()

    print(f"Model loaded: {s_model.model}")
    print(f"Max sequence length: {s_model.model.max_seq_length}")
