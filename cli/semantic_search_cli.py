import argparse
from lib.semantic_search import SemanticSearch, verify_model, embed_text, verify_embeddings

def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparser = parser.add_subparsers(dest="command", help="Available commands")

    verify_parser = subparser.add_parser("verify", help="Verify the semantic vectorization model")

    embed_parser = subparser.add_parser("embed_text", help="Embed text and get descriptive statistics for the embedding")
    embed_parser.add_argument("text", type=str, help="Text to embed")

    verify_embeddings_parser = subparser.add_parser("verify_embeddings", help="Embed text and verify embeddings in the SemanticSeach object")
    

    args = parser.parse_args()
    
    match args.command:
        case "verify":
            verify_model()
        case "embed_text":
            embed_text(args.text)
        case "verify_embeddings":
            verify_embeddings()
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()