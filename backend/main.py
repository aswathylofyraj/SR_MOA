from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

from agents.search_agent import SearchAgent
from agents.quality_agent import QualityAgent
from agents.review_agent import ReviewAgent
from agents.time_filter_agent import TimeFilterAgent
from agents.title_agent import TitleScreeningAgent
from agents.abstract_agent import AbstractScreeningAgent

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class ReviewRequest(BaseModel):
    title: str
    inclusion_criteria: List[str]
    exclusion_criteria: List[str]
    min_year: Optional[int] = None


search_agent = SearchAgent()
time_filter_agent = TimeFilterAgent()
quality_agent = QualityAgent()
review_agent = ReviewAgent()


title_agent = TitleScreeningAgent()  
abstract_agent = AbstractScreeningAgent()  

@app.post("/generate-review")
def generate_review(request: ReviewRequest):
    # Step 1: Search
    papers = search_agent.search_papers(
        title=request.title,
        inclusion_criteria=request.inclusion_criteria,
        limit=50
    )

    # Step 2: Time Filter
    if request.min_year:
        papers = time_filter_agent.filter_by_year(
            papers=papers,
            min_year=request.min_year
        )

    # Step 3: Title Screening
    title_screened_papers = title_agent.predict_inclusion(papers, request.inclusion_criteria, request.exclusion_criteria)
    if not title_screened_papers:
        return {
            "message": "No papers passed title screening.",
            "papers": [],
            "review_summary": ""
        }

    # Step 4: Abstract Screening
    abstract_screened_papers = abstract_agent.predict_inclusion(title_screened_papers, request.inclusion_criteria, request.exclusion_criteria)
    if not abstract_screened_papers:
        return {
            "message": "No papers passed abstract screening.",
            "papers": [],
            "review_summary": ""
        }

    # Step 5: Score
    scored_papers = quality_agent.score_papers(abstract_screened_papers)

    # Step 6: Review Generation
    review_summary = review_agent.generate_summary(scored_papers)

    return {
        "message": f"Generated review from {len(scored_papers)} high-quality papers.",
        "papers": scored_papers,
        "review_summary": review_summary
    }
