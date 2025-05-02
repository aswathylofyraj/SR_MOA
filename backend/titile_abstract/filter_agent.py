class FilterAgent:
    def __init__(self):
        pass  # No setup needed for now

    def filter_papers(self, papers: list, exclusion_criteria: list):
        filtered = []
        for paper in papers:
            abstract = (paper.get("abstract") or "").lower()
            if not any(ex.lower() in abstract for ex in exclusion_criteria):
                filtered.append({
                    "title": paper.get("title"),
                    "abstract": paper.get("abstract"),
                    "url": paper.get("url")
                })

        print(f"[FilterAgent] Filtered down to {len(filtered)} papers")
        return filtered
