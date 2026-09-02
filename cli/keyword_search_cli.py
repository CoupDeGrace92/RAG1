import argparse, json, string, pickle, sys, time
from nltk.stem import PorterStemmer
from typing import Self
from pathlib import Path
from contextlib import contextmanager
from dataclasses import dataclass

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using keywords")
    search_parser.add_argument("query", type=str, help="Search query")

    build_parser = subparsers.add_parser("build", help="Builds the inverted index")

    args = parser.parse_args()

    #Keyword Search
    with open("data/movies.json") as file:
        search_data = {}
        try:
            search_data = json.load(file)
        except FileNotFoundError:
            print(f"File not found to load movies: data/movies.json")
        except json.JSONDecodeError:
            print(f"JSON decode error when trying to load movie file: data/movies.json")
        except Exception as e:
            print(f"Unexpected error occured: {type(e).__name__} - {e}")

    match args.command:
        case "build":
            idx = build_command("data/movies.json")
            merida_search = idx.index.get("merida")
            if merida_search is not None:
                docs = list(merida_search)
                print(f"First document for token 'merida' = {docs[0]}")
            else:
                print("No docs matched the term merida")
            pass
        case "search":
            print(f"Searching for: {args.query}")

            q_params = tokenize_params(args.query)
            found = []
            movie_list = search_data.get("movies")

            if movie_list is not None:
                for movie in movie_list:
                    title = movie.get("title")

                    cleaned_title = remove_punc(title).lower()
                    for param in q_params:
                        if param in cleaned_title:
                            found.append(title)
                            break

                i = 1
                for title in found:
                    print(f"{i}. {title}")
                    i += 1
                    if i > 5:
                        break
                pass
        case _:
            parser.print_help()




#Helper function to remove punctuation/stopwords/create individual search params
#create punc table once globally to prevent having to construct it each call
def tokenize_params(raw: str) -> list:
    raw = remove_stopwords(raw)
    raw = remove_punc(raw)
    result = raw.lower().split()
    result = stem_words(result)
    return result

punc_table = str.maketrans("", "", string.punctuation)
def remove_punc(to_clean: str) -> str:
    return to_clean.translate(punc_table)

stopwords = set()
try:
    with open("data/stopwords.txt", "r") as file:
        stopwords = set(file.read().splitlines())
except Exception as e:
    print(f"Unexpected error occured: {type(e).__name__} - {e}")
    print("Using no filter to clean stopwords - search results will be impacted.")
def remove_stopwords(raw: str) -> str:
    raw_list = raw.split()
    cleaned = " ".join([word for word in raw_list if word not in stopwords])
        
    return cleaned

stemmer = PorterStemmer()
def stem_words(raw: list) -> list:
    cleaned = []
    for token in raw:
        stem = stemmer.stem(token)
        cleaned.append(stem)
    return cleaned

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

    def __add_document(self, doc_id, text):
        tokens = tokenize_params(text)
        for token in tokens:
            if token in self.index:
                ids = self.index.get(token)
                if ids is not None:  #This will be true since token is already in self.index with an id set
                    ids.add(doc_id)
                    self.index[token] = ids

            else:
                self.index[token] = {doc_id}

    def get_documents(self, term) -> list[str]:
        results = []
        doc_ids = self.index.get(term)

        if doc_ids is None:
            return results
        else:
            results = sorted(list(doc_ids))
        return results 

    def build(self, fp: str):
        #The movie should be in the JSON format with fields:
        #   -Title
        #   -id
        #   -description
        movies = []
        try:
            movies = load_movies(fp)
        except Exception as e:
            print(f"Error building the movie caches: {e}")
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
        
        cache_dir.mkdir(parents=True, exist_ok=True)
        idx_fp.touch(exist_ok=True)
        docmap_fp.touch(exist_ok=True)
        
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

def build_command(fp: str) -> InvertedIndex:
    idx = InvertedIndex()
    idx.build(fp)
    idx.save()
    return idx


#Helper to pin down performance problems
@dataclass
class TimingEvent:
    label: str
    start: float
    elapsed: float

class timed_events:
    def __init__(self, events=[]):
        self.events=[]

    def add_event(self: Self, event: TimingEvent):
        pass

    def view_in_order(self):
        pass

    def max(self):
        pass

    def sort(self):
        pass

    def dump_logs(self):
        cache_dir = Path("./cache")
        log_fp = cache_dir / "timing_logs.pkl"
        cache_dir.mkdir(parents=True, exist_ok=True)
        log_fp.touch(exist_ok=True)

        try:
            with open(log_fp, "wb") as file:
                pickle.dump(self.events, file)
        except Exception as e:
            print(f"Error writing to cache: {type(e).__name__} - {e}")

@contextmanager
def timed(label: str, events: timed_events):
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    events.add_event(TimingEvent(label, start, elapsed))
'''
USAGE:
with timed("load_movies", event_log):
    movies = load_movies(fp)

Use a with loop to wrap the portions of the function we want timed
Make sure the event log has been initialized
'''

if __name__ == "__main__":
    main()