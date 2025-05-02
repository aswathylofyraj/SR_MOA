class QualityAgent:
    def score_papers(self, papers: list) -> list:
        scored = []

        for paper in papers:
            abstract = paper.get("abstract", "")
            score = 0

            if abstract:
                normalized_length = min(len(abstract) / 1000, 1.0)  # Max 1.0
                score = normalized_length * 10  # Final score out of 10

            scored.append({
                "title": paper.get("title"),
                "abstract": abstract,
                "url": paper.get("url"),
                "score": round(score, 2)
            })

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored
