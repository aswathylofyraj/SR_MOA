import httpx
import os
from dotenv import load_dotenv

# Load .env file from the current directory (agents folder)
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

class TitleScreeningAgent:
    def __init__(self):
        self.api_key = os.getenv("TITLE_AGENT_API_KEY")
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"
        self.model = "deepseek/deepseek-r1:free"

    def _generate_prompt(self, titles, inclusion_criteria, exclusion_criteria):
        prompt = (
            "You are an academic paper screening assistant. Your task is to evaluate the relevance of the following paper titles for a systematic review.\n"
            "Inclusion criteria: " + ", ".join(inclusion_criteria) + "\n"
            "Exclusion criteria: " + ", ".join(exclusion_criteria) + "\n"
            "For each title below, decide whether the paper should be included in the review based on its relevance to the research topic.\n"
            "Respond for each with ONLY 'YES' or 'NO', followed by a brief reason for your decision.\n\n"
        )
        for i, title in enumerate(titles, 1):
            prompt += f"{i}. Title: {title}\nInclude? Answer:\n"
        return prompt

    def predict_inclusion(self, papers, inclusion_criteria, exclusion_criteria):
        titles = [paper.get("title", "") for paper in papers]
        prompt = self._generate_prompt(titles, inclusion_criteria, exclusion_criteria)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2
        }

        response = httpx.post(self.api_url, headers=headers, json=payload)

        if response.status_code != 200:
            raise Exception(f"Error: {response.status_code}\n{response.text}")

        reply = response.json()["choices"][0]["message"]["content"]
        lines = reply.split("\n")

        included_papers = []
        title_idx = 0

        for line in lines:
            if line.strip().startswith(str(title_idx + 1) + "."):
                if "YES" in line.upper():
                    included_papers.append(papers[title_idx])
                title_idx += 1

        if not included_papers and any("YES" in l.upper() for l in lines):
            included_papers = [paper for i, paper in enumerate(papers) if "YES" in lines[i].upper()]

        return included_papers
