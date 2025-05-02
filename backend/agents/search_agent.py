import requests
import feedparser

class SearchAgent:
    def __init__(self):
        self.s2_api_url = "https://api.semanticscholar.org/graph/v1/paper/search"
        self.arxiv_api_url = "http://export.arxiv.org/api/query"

    def search_papers(self, title: str, inclusion_criteria: list, limit: int = 10):
        query = title + " " + " ".join(inclusion_criteria)

        print(f"[SearchAgent] Searching Semantic Scholar with query: {query}")
        s2_results = self.search_semantic_scholar(query, limit=limit)

        print(f"[SearchAgent] Searching arXiv with query: {query}")
        arxiv_results = self.search_arxiv(query, limit=limit)

        # Combine results from both sources
        return s2_results + arxiv_results

    def search_semantic_scholar(self, query: str, limit: int = 10):
        params = {
            "query": query,
            "fields": "title,abstract,url",
            "limit": limit
        }

        response = requests.get(self.s2_api_url, params=params)

        if response.status_code != 200:
            print("[SearchAgent] Semantic Scholar fetch failed")
            return []

        raw_data = response.json().get("data", [])

        # Normalize format to match arXiv
        papers = []
        for paper in raw_data:
            papers.append({
                "title": paper.get("title", ""),
                "abstract": paper.get("abstract", ""),
                "url": paper.get("url", "")
            })

        return papers

    def search_arxiv(self, query: str, limit: int = 10):
        params = {
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": limit
        }

        response = requests.get(self.arxiv_api_url, params=params)
        feed = feedparser.parse(response.text)

        papers = []
        for entry in feed.entries:
            papers.append({
                "title": entry.title,
                "abstract": entry.summary,
                "url": entry.link
            })

        return papers
