import httpx
import os
import requests
from dotenv import load_dotenv

# Load environment variables from .env file in the same directory
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

class ReviewAgent:
    def __init__(self):
        self.api_key = os.getenv("REVIEW_AGENT_API_KEY")
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"
        self.model = "mistralai/mistral-7b-instruct"

    def chat_completion(self, messages):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model,
            "messages": messages
        }

        response = httpx.post(self.api_url, headers=headers, json=payload, timeout=60.0)  # 60 seconds
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    def generate_summary(self, papers: list) -> str:
        if not papers:
            return "No relevant papers were found to generate a review."

        all_papers_text = ""

        for idx, paper in enumerate(papers, start=1):
            title = paper.get("title", "No title")
            abstract = paper.get("abstract", "No abstract")
            authors = paper.get("authors", [])
            year = paper.get("year", "n.d.")

            if authors:
                first_author = authors[0].get("name", "Anon")
                if len(authors) > 1:
                    author_str = f"{first_author} et al."
                else:
                    author_str = first_author
            else:
                author_str = "Anon"

            all_papers_text += f"Title: {title}\nAuthor(s): {author_str}, {year}\nAbstract: {abstract}\n\n"

        prompt = f"""
You are an expert academic assistant. Based on the following research paper titles and abstracts, write a single, professional summary that can be directly used in a thesis or academic report.

Your output must:
- Be formal, academic, and clearly structured
- Identify common themes, innovations, global insights, and gaps in research
- Synthesize the findings across all papers instead of listing them individually
- Do not include in-text citation numbers or a references section
- Write summary with proper heading

Here are the papers:

{all_papers_text}

Now write the summary without in-text citations or references.
"""

        messages = [
            {"role": "system", "content": "You are a scholarly assistant that writes high-quality summaries."},
            {"role": "user", "content": prompt}
        ]

        return self.chat_completion(messages)

"""def test_review_agent():
    agent = ReviewAgent()

    # Example test input: a couple of simplified papers
    papers = [
        {
            "title": "Deep Learning in Breast Cancer Diagnosis",
            "abstract": "This paper explores deep learning approaches for breast cancer detection using medical imaging datasets.",
            "authors": [{"name": "Jane Doe"}, {"name": "John Smith"}],
            "year": "2023"
        },
        {
            "title": "AI-Based Lung Cancer Screening Techniques",
            "abstract": "An overview of AI applications in the early detection of lung cancer, focusing on CT scan analysis and pattern recognition.",
            "authors": [{"name": "Alice Brown"}],
            "year": "2022"
        }
    ]

    try:
        summary = agent.generate_summary(papers)
        print("Generated Summary:\n")
        print(summary)
    except Exception as e:
        print(f"Error generating review summary: {e}")
test_review_agent() """
