import argparse, os, time, json
from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat import ChatCompletionUserMessageParam

from utils.search_utils import max_min_normalize, load_movies, DocumentObject
from utils.hybrid_search_utils import HybridSearch

from sentence_transformers import CrossEncoder

def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    normalize = subparsers.add_parser("normalize",  help="Normalize a list of scores relative to the min/max")
    normalize.add_argument("list", type=float, nargs="*", help="list to normalize")

    weighted_search = subparsers.add_parser("weighted-search", help="Hybrid weighted search between BM25 keyword matching and semantic searching")
    weighted_search.add_argument("query", type=str, help="Query to search for")
    weighted_search.add_argument("--alpha", type=float, default=05., help="Alpha value between 0 and 1.  1 puts entire weight on semantic")
    weighted_search.add_argument("--limit", type=int, default=5, help="Maximum search results returned")

    rrf_search = subparsers.add_parser("rrf-search", help="RRF search hybrid between BM25 keyword matching and semantic search")
    rrf_search.add_argument("query", type=str, help="Query to search for")
    rrf_search.add_argument("-k", type=int, default=60, help="Weight given to higher-ranked results - lower gives more weight to higher-ranked results")
    rrf_search.add_argument("--limit", type=int, default=5, help="Maximum search results returned")
    rrf_search.add_argument("--enhance", type=str, choices=["spell", "rewrite", "expand"], help="Query enhancement method")
    rrf_search.add_argument("--rerank-method", type=str, choices=["individual", "batch", "cross_encoder"], help="Method for nuanced reranking using an LLM")

    args = parser.parse_args()

    match args.command:
        case "normalize":
            norm = max_min_normalize(args.list)
            for item in norm:
                print(f"{item}:.4f")
        case "weighted-search":
            hs = HybridSearch(load_movies("./data/movies.json"))
            results = hs.weighted_search(args.query, args.alpha, args.limit)
            for idx, item in enumerate(results):
                doc = item.get("document")
                if isinstance(doc, str):
                    if len(doc) > 100:
                        doc = doc[:100]
                print(f"{idx+1}. {item.get("title")}")
                print(f"  Hybrid Score: {item.get("weighted"):.3f}")
                print(f"  BM25: {item.get("keyword_score"):.3f}, Semantic: {item.get("semantic_score"):.3f}")
                print(f"{doc}...")
        case "rrf-search":
            hs = HybridSearch(load_movies("./data/movies.json"))
            q: str = args.query
            load_dotenv()
            api_key = os.environ.get("OPENROUTER_API_KEY")
            if not api_key:
                raise RuntimeError("OPENROUTER_API_KEY environment variable not set")

            client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=api_key
            )
            match args.enhance:
                case "spell":

                    messages: list[ChatCompletionUserMessageParam] = [
                        {
                            "role": "user",
                            "content": f"""
                            Fix any spelling errors in the user-provided movie search query below.
                            Correct only clear, high-confidence typos. Do not rewrite, add, remove, or reorder words.
                            Preserve punctuation and capitalization unless a change is required for a typo fix.
                            If there are no spelling errors, or if you're unsure, output the original query unchanged.
                            Output only the final query text, nothing else.
                            User query: "{q}"
                            """
                        }
                    ]

                    response = client.chat.completions.create(
                        model="openrouter/free",
                        messages=messages
                    )
                    if response.choices[0].message.content:
                        q = response.choices[0].message.content
                    print(f"Enhanced query ({args.enhance}): '{args.query}' -> '{q}'")

                case "rewrite":
                    messages: list[ChatCompletionUserMessageParam] = [
                        {
                            "role": "user",
                            "content": f"""
                            Rewrite the user-provided movie search query below to be more specific and searchable.

                            Consider:
                            - Common movie knowledge (famous actors, popular films)
                            - Genre conventions (horror = scary, animation = cartoon)
                            - Keep the rewritten query concise (under 10 words)
                            - It should be a Google-style search query, specific enough to yield relevant results
                            - Don't use boolean logic

                            Examples:
                            - "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"
                            - "movie about bear in london with marmalade" -> "Paddington London marmalade"
                            - "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"

                            If you cannot improve the query, output the original unchanged.
                            Output only the rewritten query text, nothing else.

                            User query: "{q}"
                            """
                        }
                    ]

                    response = client.chat.completions.create(
                        model="openrouter/free",
                        messages=messages
                    )
                    if response.choices[0].message.content:
                        q = response.choices[0].message.content
                    print(f"Enhanced query ({args.enhance}): '{args.query}' -> '{q}'")

                case "expand":
                    messages: list[ChatCompletionUserMessageParam] = [
                        {
                            "role": "user",
                            "content": f"""
                            Expand the user-provided movie search query below with related terms.

                            Add synonyms and related concepts that might appear in movie descriptions.
                            Keep expansions relevant and focused.
                            Output only the additional terms; they will be appended to the original query.

                            Examples:
                            - "scary bear movie" -> "scary horror grizzly bear movie terrifying film"
                            - "action movie with bear" -> "action thriller bear chase fight adventure"
                            - "comedy with bear" -> "comedy funny bear humor lighthearted"

                            User query: "{q}"
                            """
                        }
                    ]

                    response = client.chat.completions.create(
                        model="openrouter/free",
                        messages=messages
                    )
                    if response.choices[0].message.content:
                        q = q + ' ' + response.choices[0].message.content
                    print(f"Enhanced query ({args.enhance}): '{args.query}' -> '{q}'")
                                    
            
            if args.rerank_method:
                results = hs.rrf_search(q, args.k, args.limit * 5)
                match args.rerank_method:
                    case "individual":
                        reranked = []
                        for idx, item in enumerate(results):
                            rerank: float = 0
                            doc = item.get("document")
                            messages: list[ChatCompletionUserMessageParam] = []
                            if doc:
                                messages: list[ChatCompletionUserMessageParam] = [
                                    {
                                        "role": "user",
                                        "content": f"""
                                        Rate how well this movie matches the search query.

                                        Query: "{q}"
                                        Movie: {item.get("title", "")} - {item.get("document", "")}

                                        Consider:
                                        - Direct relevance to query
                                        - User intent (what they're looking for)
                                        - Content appropriateness

                                        Rate 0-10 (10 = perfect match).
                                        Output ONLY the number in your response, no other text or explanation.

                                        Score:
                                        """
                                    }
                                ]
                            response = client.chat.completions.create(
                                model="openrouter/free",
                                messages=messages
                            )
                            if response.choices[0].message.content:
                                try:
                                    rerank = float(response.choices[0].message.content)
                                except ValueError:
                                    print(f"Could not convert LLM rerank into a float: {response.choices[0].message.content}")
                            item["rerank"] = rerank
                            reranked.append(item)
                            if idx != len(results) - 1:
                                time.sleep(3)
                        reranked.sort(key=lambda item: item["rerank"], reverse=True)
                        results = reranked[:args.limit]
                        for idx, item in enumerate(results):
                            doc = item.get("document")
                            if isinstance(doc, str):
                                if len(doc)>100:
                                    doc = doc[:100]
                            print(f"{idx+1}. {item.get("title")}")
                            print(f"  Re-rank Score: {item.get("rerank:") or 0}")
                            print(f"  RRF Score: {item.get("rrf")}")
                            print(f"  BM25 Rank: {item.get("bm_rank") or 0}, Semantic Rank: {item.get("sem_rank") or 0}")
                            print(f"  {doc}...")
                    case "batch":
                        doc_list_str = json.dumps(results)
                        messages=[
                            {
                                "role":"user",
                                "content": f"""
                                Rank the movies listed below by relevance to the following search query.

                                Query: "{q}"

                                Movies:
                                {doc_list_str}

                                Return the movie IDs in order of relevance, best match first.

                                Your response must be a raw JSON array of integers.
                                Do not wrap the JSON in Markdown. Do not use a ```json code block.
                                Do not include any explanatory text.

                                For example:
                                [75, 12, 34, 2, 1]

                                Ranking:
                                """
                            }
                        ]
                        response = client.chat.completions.create(
                            model="openrouter/free",
                            messages=messages
                        )
                        print(response.choices[0].message.content)
                        relevance_list = []
                        if response.choices[0].message.content:
                            relevance_list = json.loads(response.choices[0].message.content)

                        if len(results) < len(relevance_list):
                            raise ImportError(f"LLM result list larger than input list - RESULTS SIZE {len(results)} - RELEVANCE SIZE {len(relevance_list)}")

                        results_by_id = {item.get("id"): item for item in results}

                        out = []
                        for movie_id in relevance_list:
                            item = results_by_id.get(movie_id)
                            if item is None:
                                raise ValueError(f"LLM returned unknown movie ID: {movie_id}")
                            out.append(item)
                        out = out[:args.limit]
                        print(f"Re-ranking top {args.limit} results using batch method...")
                        print(f"Reciprocal Rank Fusion Results for '{args.query}'")
                        print()
                        for idx, item in enumerate(out):
                            doc = item.get("document")
                            if isinstance(doc, str):
                                if len(doc)>100:
                                    doc = doc[:100]
                            print(f"{idx+1}. {item.get("title")}")
                            print(f"  Re-rank: {idx + 1}")
                            print(f"  RRF Score: {item.get("rrf")}")
                            print(f"  BM25 Rank: {item.get("bm_rank") or 0}, Semantic Rank: {item.get("sem_rank") or 0}")
                            print(f"  {doc}...")
                    case "cross_encoder":
                        pairs = []
                        for item in results:
                            pairs.append([q, f"{item.get('title', '')} - {item.get('document', '')}"])

                        cross_encoder = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L2-v2")
                        scores = cross_encoder.predict(pairs)

                        for idx, item in enumerate(results):
                            item["cross"] = scores[idx]

                        results = sorted(results, key=lambda x: x["cross"])

                        for idx, item in enumerate(results):
                            doc = item.get("document")
                            if isinstance(doc, str):
                                if len(doc)>100:
                                    doc = doc[:100]

                            print(f"{idx+1}. {item.get("title")}")
                            print(f"Cross Encoder Score: {item.get("cross")}")
                            print(f"  RRF Score: {item.get("rrf")}")
                            print(f"  BM25 Rank: {item.get("bm_rank") or 0}, Semantic Rank: {item.get("sem_rank") or 0}")
                            print(f"  {doc}...")

            else:
                results = hs.rrf_search(q, args.k, args.limit)
                for idx, item in enumerate(results):
                    doc = item.get("document")
                    if isinstance(doc, str):
                        if len(doc)>100:
                            doc = doc[:100]
                    print(f"{idx+1}. {item.get("title")}")
                    print(f"  RRF Score: {item.get("rrf")}")
                    print(f"  BM25 Rank: {item.get("bm_rank") or 0}, Semantic Rank: {item.get("sem_rank") or 0}")
                    print(f"  {doc}...")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()