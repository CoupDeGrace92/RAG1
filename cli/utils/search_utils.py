import string, pickle, math, json
from nltk.stem import PorterStemmer
from typing import Self, Counter
from pathlib import Path
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
from collections import Counter
from constants import BM25_k1, BM25_B

from timing_logs.data_structures import event_log, timed

#Helper function to remove punctuation/stopwords/create individual search params
#create punc table once globally to prevent having to construct it each call
def tokenize_params(raw: str) -> list:
    raw = raw.lower()
    raw = remove_punc(raw)
    raw = remove_stopwords(raw)
    result = raw.split()
    result = stem_words(result)
    return result

def tokenize_single(raw: str) -> str:
    cleaned = tokenize_params(raw)
    if len(cleaned) > 1 or len(cleaned) == 0:
        raise ValueError("Tokenize only a single parameter")
    return cleaned[0]

punc_table = str.maketrans("", "", string.punctuation)
def remove_punc(to_clean: str) -> str:
    return to_clean.translate(punc_table)

stopwords = set()
try:
    with open("data/stopwords.txt", "r") as file:
        text = file.read()
        text = text.lower()
        text = remove_punc(text)
        stopwords = set(text.splitlines())
except Exception as e:
    print(f"Unexpected error occured: {type(e).__name__} - {e}")
    print("Using no filter to clean stopwords - search results will be impacted.")

def remove_stopwords(raw: str) -> str:
    raw_list = raw.split()
    cleaned = " ".join([word for word in raw_list if word not in stopwords])
    return cleaned

stemmer = PorterStemmer()
@lru_cache(maxsize=500)
def stem_word(token: str) -> str:
    return stemmer.stem(token)

def stem_words(raw: list) -> list:
    return [stem_word(token) for token in raw]

#We load movies assuming they are in a specific JSON format
def load_movies(fp: str) -> list[dict]:
    result = []
    data = {}
    with open(fp) as file:
        try:
            data = json.load(file)
        except FileNotFoundError:
            print(f"File not found to load movies: {fp}")
            return result
        except json.JSONDecodeError:
            print(f"JSON decode error when trying to load movie file: {fp}")
            return result
        except Exception as e:
            print(f"Unexpected error occured: {type(e).__name__} - {e}")
            return result

    movies = data.get("movies")
    if movies is not None:
        result = movies

    return result #The typing assumes the JSON is formatted how we expect


class DocumentObject:
    def __init__(self, title = None, id: str | None = None, description = None):
        self.id = id
        self.title = title
        self.description = description

    def build_from_dict(self, d: dict) -> Self:
        for key in ["id", "title", "description"]:
            if d.get(key) is not None:
                setattr(self, key, d[key])
        
        return self


class InvertedIndex:
    def __init__(self):
        self.index: dict[str, set[str]] = {}  #token to set of document id
        self.docmap: dict[str, DocumentObject] = {}  #doc id to full doc object
        #We can consider making the dict into a class instead 
        self.term_frequencies: dict[str, Counter] = {} #Doc id to a counter dict for #appearances of each token in that doc
        self.doc_lengths = {} #doc id to length of the document

    def __add_document(self: Self, doc_id: str, text: str) -> None:
        tokens = tokenize_params(text)
        if self.term_frequencies.get(doc_id):
            return
        self.term_frequencies[doc_id] = Counter()
        self.doc_lengths[doc_id] = len(tokens)
        for token in tokens:
            if token in self.index:
                ids = self.index.get(token)
                if ids is not None:  #This will be true since token is already in self.index with an id set
                    ids.add(doc_id)
                    self.index[token] = ids

            else:
                self.index[token] = {doc_id}

            self.term_frequencies[doc_id].update([token])

    def __get_avg_doc_length(self) -> float:
        if len(self.doc_lengths) == 0:
            return 0.0

        total = 0
        for doc in self.doc_lengths:
            total += self.doc_lengths[doc]
        return total/len(self.doc_lengths)

    def get_tf(self, doc_id, term):
        return self.term_frequencies[doc_id][term]

    def get_idf(self: Self, term: str) -> float:
        tokenized_term = tokenize_single(term)
        doc_count = len(self.docmap)
        match_count = len(self.index[tokenized_term])
        return math.log((doc_count + 1) / (match_count + 1))

    def get_bm25_idf(self: Self, term: str) -> float:
        tokenized_term = tokenize_single(term)
        doc_count = len(self.docmap)
        match_count = len(self.index[tokenized_term])
        return math.log((doc_count - match_count + 0.5) / (match_count + 0.5) + 1)

    def get_bm25_tf(self, doc_id: int, term: str, k1: float=BM25_k1, b: float=BM25_B) -> float:
        tokenized = tokenize_single(term)
        raw_tf = self.get_tf(doc_id, tokenized)
        doc_length = self.doc_lengths[doc_id]
        avg_doc_length = self.__get_avg_doc_length()
        length_norm = 1 - b + b *(doc_length / avg_doc_length)
        return (raw_tf * (k1 + 1)) / (raw_tf + k1 *length_norm)

    def get_documents(self, term) -> list[str]:
        results = []
        doc_ids = self.index.get(term)

        if doc_ids is None:
            return results
        else:
            results = sorted(list(doc_ids))
        return results 

    def bm25(self, doc_id, term) -> float:
        tf = self.get_bm25_tf(doc_id, term)
        idf = self.get_bm25_idf(term)
        return tf * idf

    def search(self, params: list[str], limit: int):
        results = []
        seen = set()
        for param in params:
            for doc_id in self.get_documents(param):
                if doc_id in seen:
                    continue

                seen.add(doc_id)
                results.append(self.docmap[doc_id])

                if len(results) >= limit:
                    return results

    def bm25_search(self, query: str, limit: int):
        tokens = tokenize_params(query)
        scores = {}
        for doc in self.docmap:
            BM25 = 0
            for token in tokens:
                BM25 += self.bm25(doc, token)
            scores[doc] = BM25
        s = sorted(scores.items(), key=lambda item: item[1], reverse=True)

        out = []
        i=0
        for score in s:
            out.append((self.docmap[score[0]], score[1]))
            i += 1
            if i >= limit:
                break
        return out


    def build(self, fp: str):
        #The movie should be in the JSON format with fields:
        #   -Title
        #   -id
        #   -description
        movies = []
        with timed("load_movies", event_log):
            try:
                movies = load_movies(fp)
            except Exception as e:
                print(f"Error building the movie caches: {e}")

        with timed("main_build_loop", event_log):
            for movie in movies:
                m = DocumentObject().build_from_dict(movie)

                if m.title == None or m.id == None:
                    continue
                desc = m.description
                if desc is None:
                    desc = ""

                self.__add_document(m.id, f"{m.title} {desc}")
                self.docmap[m.id] = m

        return self #allowing for method chaining later if we want it
        
    def save(self):
        cache_dir = Path("./cache")
        idx_fp = cache_dir / "index.pkl"
        docmap_fp = cache_dir / "docmap.pkl"
        counter_fp = cache_dir / "term_frequencies.pkl"
        doc_lengths_fp = cache_dir / "doc_lengths.pkl"
        
        cache_dir.mkdir(parents=True, exist_ok=True)
        idx_fp.touch(exist_ok=True)
        docmap_fp.touch(exist_ok=True)
        counter_fp.touch(exist_ok=True)
        doc_lengths_fp.touch(exist_ok=True)
        
        try:
            with open(docmap_fp, "wb") as file:
                pickle.dump(self.docmap, file)
        except Exception as e:
            print(f"Error writing to cache: {type(e).__name__} - {e}")
            
        try:
            with open(idx_fp, "wb") as file:
                pickle.dump(self.index, file)
        except Exception as e:
            print(f"Error writing to cache: {type(e).__name__} - {e}")

        try:
            with open(counter_fp, "wb") as file:
                pickle.dump(self.term_frequencies, file)
        except Exception as e:
            print(f"Error writing to counter cache: {type(e).__name__} - {e}")

        try:
            with open(doc_lengths_fp, "wb") as file:
                pickle.dump(self.doc_lengths, file)
        except Exception as e:
            print(f"Error writing to doc lengths cache: {type(e).__name__} - {e}")


    def load(self):
        cache_dir = Path("./cache")
        idx_fp = cache_dir / "index.pkl"
        docmap_fp = cache_dir / "docmap.pkl"
        counter_fp = cache_dir / "term_frequencies.pkl"
        doc_lengths_fp = cache_dir / "doc_lengths.pkl"
        #Docmap first
        docmap = {}
        try:
            with open(docmap_fp, "rb") as file:
                docmap = pickle.load(file)
        except Exception as e:
            print(f"Error reading from docmap cache: {type(e).__name__} - {e}")

        index = {}
        try:
            with open(idx_fp, "rb") as file:
                index = pickle.load(file)
        except Exception as e:
            print(f"Error reading from index cache: {type(e).__name__} - {e}")

        t_f = {}
        try:
            with open(counter_fp, "rb") as file:
                t_f = pickle.load(file)
        except Exception as e:
            print(f"Error reading from counter cache: {type(e).__name__} - {e}")

        doc_l = {}
        try:
            with open(doc_lengths_fp, "rb") as file:
                doc_l = pickle.load(file)
        except Exception as e:
            print(f"Error reading from counter cache: {type(e).__name__} - {e}")

        if docmap:
            self.docmap = docmap
        if index:
            self.index = index
        if t_f:
            self.term_frequencies = t_f
        if doc_l:
            self.doc_lengths = doc_l


def build_command(fp: str) -> InvertedIndex:
    idx = InvertedIndex()
    idx.build(fp)
    idx.save()
    return idx
