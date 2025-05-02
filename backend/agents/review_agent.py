import os
import requests

class ReviewAgent:
    def __init__(self):
        self.api_key = "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"
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

        response = requests.post(self.api_url, headers=headers, json=payload)
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
- write summary with proper heading
Here are the papers:

{all_papers_text}

Now write the summary without in-text citations or references.
"""

        messages = [
            {"role": "system", "content": "You are a scholarly assistant that writes high-quality summaries."},
            {"role": "user", "content": prompt}
        ]

        return self.chat_completion(messages)