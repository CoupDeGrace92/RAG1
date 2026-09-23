import argparse, shutil
from lib.semantic_search import SemanticSearch, verify_model, embed_text, verify_embeddings, embed_text_query, ChunkedSemanticSearch
from utils.search_utils import load_movies, chunk_print, semantic_chunk_print
from timing_logs.data_structures import event_log, timed

def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparser = parser.add_subparsers(dest="command", help="Available commands")

    verify_parser = subparser.add_parser("verify", help="Verify the semantic vectorization model")

    embed_parser = subparser.add_parser("embed_text", help="Embed text and get descriptive statistics for the embedding")
    embed_parser.add_argument("text", type=str, help="Text to embed")

    verify_embeddings_parser = subparser.add_parser("verify_embeddings", help="Embed text and verify embeddings in the SemanticSeach object")

    embed_query_parser = subparser.add_parser("embed_query", help="Embed query and get descriptive statistics for the embedding")
    embed_query_parser.add_argument("query", type=str, help="Query to embed")

    search_parser = subparser.add_parser("search", help="Semantic search for query")
    search_parser.add_argument("query", type=str, help="Query to search for")
    search_parser.add_argument("--limit", type=int, default=5, help="Maximum number of results to return")

    chunk_print_parser = subparser.add_parser("chunk", help="Chunk text into smaller segments and print it to the terminal")
    chunk_print_parser.add_argument("text", type=str, help="Text to be chunked")
    chunk_print_parser.add_argument("--chunk-size", type=int, default=200, help="Number of words maximum in each chunk")
    chunk_print_parser.add_argument("--overlap", type=int, default=0, help="Number of words that overlap the previous chunk")

    semantic_chunk_parser = subparser.add_parser("semantic_chunk", help="Chunk text by sentances and print to terminal")
    semantic_chunk_parser.add_argument("text", type=str, help="The text to chunk")
    semantic_chunk_parser.add_argument("--overlap", type=int, default=0, help="The number of overlaping sentances in subsequent chunks")
    semantic_chunk_parser.add_argument("--max-chunk-size", type=int, default=4, help="Maximum number of sentances the chunk contains")

    embed_chunks_parser = subparser.add_parser("embed_chunks", help="Create chunk embedding and get descriptive statistics")

    search_chunked_parser = subparser.add_parser("search_chunked", help="Search for a query using semantic searching over chunked descriptions")
    search_chunked_parser.add_argument("query", type=str, help="query to search for")
    search_chunked_parser.add_argument("--limit", type=int, default=5, help="Maximum results to return")

    args = parser.parse_args()
    
    match args.command:
        case "verify":
            verify_model()
        case "embed_text":
            embed_text(args.text)
        case "verify_embeddings":
            verify_embeddings()
        case "embed_query":
            embed_text_query(args.query)
        case "search":
            s_search = SemanticSearch()
            movie_fp = "./data/movies.json"
            documents = load_movies(movie_fp)
            s_search.load_or_create_embeddings(documents)
            results  = s_search.search(args.query, args.limit)
            i=1
            width, _ = shutil.get_terminal_size() 
            width = width - 3
            print(f"Searching {args.query}")
            for result in results:
                desc = result.get("description")
                if len(desc) > width + 1:
                    desc = desc[:width] + "..."
                print(f"{i}. {result.get("title")}\n  {desc}")
                i+=1
        case "chunk":
            chunk_print(args.text, args.chunk_size, args.overlap)
        case "semantic_chunk":
            semantic_chunk_print(args.text, args.max_chunk_size, args.overlap)
        case "embed_chunks":
            c_search = ChunkedSemanticSearch()
            embeddings = c_search.build_chunk_embeddings(load_movies("./data/movies.json"))
            print(f"Generated {len(embeddings)} chunked embeddings")
        case "search_chunked":
            c_search = ChunkedSemanticSearch()
            embeddings = c_search.load_or_create_chunk_embeddings(load_movies("./data/movies.json"))
            print(f"Loaded {len(embeddings)} chunked embeddings")
            results = c_search.search_chunks(args.query, args.limit)
            for idx, movie in enumerate(results):
                print(f"\n{idx}. {movie.get("title")} (score: {movie.get("score"):.4f})")
                print(f"    {movie.get("document")}...")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()