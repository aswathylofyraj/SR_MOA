import re

class QualityAgent:
    def score_papers(self, papers: list) -> list:
        scored = []

        for paper in papers:
            title = paper.get("title", "")
            abstract = paper.get("abstract", "")
            url = paper.get("url", "")

            score = 0

            # 1. Abstract length (max 2 points)
            length_score = min(len(abstract) / 1000, 1.0) * 2
            score += length_score

            # 2. Presence of scientific structure (max 3 points)
            structure_keywords = ["background", "method", "result", "conclusion"]
            structure_score = sum(1 for kw in structure_keywords if re.search(rf"\b{kw}\b", abstract.lower()))
            structure_score = min(structure_score, 3)
            score += structure_score

            # 3. Study design keywords (max 3 points)
            quality_keywords = ["randomized", "systematic review", "meta-analysis", "double-blind", "controlled trial"]
            design_score = sum(1 for kw in quality_keywords if kw in abstract.lower())
            design_score = min(design_score, 3)
            score += design_score

            # 4. Clarity (basic check: punctuation and grammar-like features) (max 2 points)
            clarity_score = 2 if abstract.count(".") >= 3 and len(abstract.split()) >= 100 else 1 if len(abstract.split()) > 50 else 0
            score += clarity_score

            scored.append({
                "title": title,
                "abstract": abstract,
                "url": url,
                "score": round(score, 2)
            })

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored
