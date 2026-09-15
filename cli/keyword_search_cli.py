import argparse, json
from constants import BM25_k1, BM25_B
from utils.search_utils import *

#Temp imports for debugging
from timing_logs.data_structures import event_log, timed

def main() -> None:
    #Init event timer for debugging

    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using keywords")
    search_parser.add_argument("query", type=str, help="Search query")

    build_parser = subparsers.add_parser("build", help="Builds the inverted index")

    tf_parser = subparsers.add_parser("tf", help="Returns term frequency of input token")
    tf_parser.add_argument("doc_id", type=int, help="Document id to search within")
    tf_parser.add_argument("term", type=str, help="Term to search for")

    idf_parser = subparsers.add_parser("idf", help="Calculates the inverse document frequency for a given term")
    idf_parser.add_argument("term", type=str, help="Term to search for the inverted document frequency")

    tfidf_parser = subparsers.add_parser("tfidf", help="Combined term frequency/inverse document frequency measure for a given term")
    tfidf_parser.add_argument("term", type=str, help="Term to search for")

    bm25_parser = subparsers.add_parser("bm25idf", help="BM25 inverted document frequency measure")
    bm25_parser.add_argument("term", type=str, help="Term to search for")

    bm25tf_parser = subparsers.add_parser("bm25tf", help="BM25 term frequency measure for a given doc_id/term pair")
    bm25tf_parser.add_argument("doc_id", type=int, help="Document id to search within")
    bm25tf_parser.add_argument("term", type=str, help="Term to search for")
    '''
    #THIS DOES NOT ALLOW SPECIFICATION OF B WITHOUT K1
    bm25tf_parser.add_argument("k1", type=float, nargs="?", default=BM25_k1, help="Tunable BM25 K1 parameter")
    bm25tf_parser.add_argument("b", type=float, nargs="?", default=BM25_B, help="Tunable BM25 b parameter for length normalization")
    '''
    bm25tf_parser.add_argument("--k1", type=float, default=BM25_k1, help="Tunable BM25 k1 parameter")
    bm25tf_parser.add_argument("--b", type=float, default=BM25_B, help="Tunable BM25 b parameter for length normalization")

    bm25search_parser = subparsers.add_parser("bm25search", help="Search movies using full BM25 scoring")
    bm25search_parser.add_argument("query", type=str, help="Search Query")
    bm25search_parser.add_argument("--limit", type=int, default=5, help="Maximum number of results to returns")


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
            build_command("data/movies.json")
        case "search":
            i_idx = InvertedIndex()
            try:
                i_idx.load()
            except Exception as e:
                print(f"Error loading data to search from: {e}")

            print(f"Searching for: {args.query}")

            q_params = tokenize_params(args.query)

            found = i_idx.search(q_params, 5)
            if found == None:
                print("No matches found")
            else:
                i = 1
                for movie in found:
                    print(f"{i}. ID: {movie.id}  TITLE: {movie.title}")
                    i += 1
                    if i > 100:
                        break
        case "tf":
            i_idx = InvertedIndex()
            try:
                i_idx.load()
            except Exception as e:
                print(f"Error loading data to search from: {e}")

            try:
                result = i_idx.get_tf(args.doc_id, args.term)
                print(f"Document ID: {args.doc_id},  Term: {args.term},  Freq: {result}")
            except Exception as e:
                print(f"Error fetching frequency of input tokens: {type(e).__name__} - {e}")

        case "idf":
            i_idx = InvertedIndex()
            try:
                i_idx.load()
            except Exception as e:
                print(f"Error loading data to search from: {e}")

            idf = i_idx.get_idf(args.term)
            print(f"Inverse document frequency of '{args.term}':  {idf:.2f}")

        case "tfidf":
            i_idx = InvertedIndex()
            try:
                i_idx.load()
            except Exception as e:
                print(f"Error loading data to search from: {e}")

            tf = 0
            try:
                tf = i_idx.get_tf(args.doc_id, args.term)
            except Exception as e:
                print(f"Error fetching frequency of input tokens: {type(e).__name__} - {e}")

            idf = i_idx.get_idf(args.term)
            tf_idf = tf * idf
            print(f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {tf_idf:.2f}")

        case "bm25idf":
            i_idx = InvertedIndex()
            try:
                i_idx.load()
            except Exception as e:
                print(f"Error loading data to search from: {e}")

            try:
                bm25 = i_idx.get_bm25_idf(args.term)
                print(f"BM25 IDF score of '{args.term}': {bm25:.2f}")
            except Exception as e:
                print(f"Error calculating bm25: {type(e).__name__} - {e}")
        case "bm25tf":
            i_idx = InvertedIndex()
            try:
                i_idx.load()
            except Exception as e:
                print(f"Error loading data to search from: {e}")

            try:
                bm25 = i_idx.get_bm25_tf(args.doc_id, args.term, args.k1, args.b)
                print(f"BM25 TF score of '{args.term}' in document '{args.doc_id}': {bm25:.2f}")
            except Exception as e:
                print(f"Error calculating BM25 TF score: {type(e).__name__} - {e}")

        case "bm25search":
            i_idx = InvertedIndex()
            try:
                i_idx.load()
            except Exception as e:
                print(f"Error loading data to search from: {e}")

            try:
                result_tuples = i_idx.bm25_search(args.query, args.limit)
                i=1
                for tuple in result_tuples:
                    doc = tuple[0]
                    print(f"{i}. ({doc.id}) {doc.title} - Score: {tuple[1]:.2f}")
                    i+=1
            except Exception as e:
                print(f"Error in getting search results: {type(e).__name__} - {e}")

        case _:
            parser.print_help()

    event_log.dump_logs()






if __name__ == "__main__":
    main()