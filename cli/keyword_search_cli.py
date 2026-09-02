import argparse, json, string
from nltk.stem import PorterStemmer

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

            q_params = search_parms(args.query)
            found = []
            movie_list = search_data.get("movies")

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
def search_parms(raw: str) -> list:
    raw = remove_stopwords(raw)
    raw = remove_punc(raw)
    result = raw.lower().split()
    result = stem_words(result)
    return result

punc_table = str.maketrans("", "", string.punctuation)
def remove_punc(to_clean: str) -> str:
    return to_clean.translate(punc_table)


def remove_stopwords(raw: str) -> str:
    with open("data/stopwords.txt" ,"r") as file:
        stopwords = file.read().splitlines()

    raw_list = raw.split()
    cleaned = " ".join([word for word in raw_list if word not in stopwords])
        
    return cleaned

def stem_words(raw: list) -> list:
    stemmer = PorterStemmer()
    cleaned = []
    for token in raw:
        stem = stemmer.stem(token)
        cleaned.append(stem)
    return cleaned


if __name__ == "__main__":
    main()