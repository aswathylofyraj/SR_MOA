from datetime import datetime

class TimeFilterAgent:
    def filter_by_year(self, papers: list[dict], min_year: int) -> list[dict]:
        """
        Filters out papers published before the given min_year.
        Assumes each paper dict has a 'year' key (int).
        If 'year' is missing, defaults to current year.
        """
        return [
            paper for paper in papers
            if paper.get("year", datetime.now().year) >= min_year
        ]
