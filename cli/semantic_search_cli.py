import argparse
from lib.semantic_search import SemanticSearch, verify_model

def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparser = parser.add_subparsers(dest="command", help="Available commands")

    verify_parser = subparser.add_parser("verify", help="Verify the semantic vectorization model")

    args = parser.parse_args()
    
    match args.command:
        case "verify":
            verify_model()
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()