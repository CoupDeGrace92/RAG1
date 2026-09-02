import argparse, json


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using keywords")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    #Keyword Search
    with open("data/movies.json") as file:
        search_data = json.load(file)

    match args.command:
        case "search":
            print(f"Searching for: {args.query}")

            found = []
            movie_list = search_data.get("movies")

            for movie in movie_list:
                title = movie.get("title")
                if args.query in title:
                    found.append(title)

            i = 1
            for title in found:
                print(f"{i}. {title}")
                i += 1
                if i > 5:
                    break
            pass
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()