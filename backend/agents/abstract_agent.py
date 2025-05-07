import httpx
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

class AbstractScreeningAgent:
    def __init__(self):
        self.api_key = os.getenv("ABSTRACT_AGENT_API_KEY")
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"
        self.model = "deepseek/deepseek-chat"

    def _generate_prompt(self, abstracts, inclusion_criteria, exclusion_criteria):
        prompt = (
            "You are an academic paper screening assistant. Your task is to evaluate the relevance of the following abstracts for a systematic review.\n"
            "Inclusion criteria: " + ", ".join(inclusion_criteria) + "\n"
            "Exclusion criteria: " + ", ".join(exclusion_criteria) + "\n"
            "For each abstract below, decide whether the study should be included in the review based on its relevance to the research topic.\n"
            "Respond for each with ONLY 'YES' or 'NO', followed by a brief reason for your decision.\n\n"
        )
        for i, abstract in enumerate(abstracts, 1):
            prompt += f"{i}. Abstract: {abstract}\nInclude? Answer:\n"
        return prompt

    def predict_inclusion(self, papers, inclusion_criteria, exclusion_criteria, batch_size=2):
        included_papers = []

        for i in range(0, len(papers), batch_size):
            batch = papers[i:i + batch_size]
            abstracts = [paper.get("abstract", "")[:1000] for paper in batch]  # truncate long abstracts
            prompt = self._generate_prompt(abstracts, inclusion_criteria, exclusion_criteria)

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }

            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.2,
                "max_tokens": 1000
            }

            try:
                response = httpx.post(self.api_url, headers=headers, json=payload)
                response.raise_for_status()
                data = response.json()
            except Exception as e:
                print("❌ Request failed:", str(e))
                print("🔍 Response text:", response.text if 'response' in locals() else "No response received.")
                continue

            if "choices" not in data:
                print("⚠️ Unexpected response format. Full response:")
                print(data)
                continue

            reply = data["choices"][0]["message"]["content"]
            lines = reply.split("\n")
            abstract_idx = 0

            for line in lines:
                if line.strip().startswith(str(abstract_idx + 1) + "."):
                    if "YES" in line.upper():
                        included_papers.append(batch[abstract_idx])
                    abstract_idx += 1

        return included_papers

def test_abstract_screening_agent():
    agent = AbstractScreeningAgent()

    papers = [
        {
            "title": "Artificial Intelligence for Cancer Diagnosis",
            "abstract": "This study explores how AI models can assist in diagnosing cancer with high accuracy..."
        },
        {
            "title": "A Review of Agricultural Robotics",
            "abstract": "This paper presents an overview of robotics used in agriculture rather than in clinical settings..."
        }
    ]

    inclusion_criteria = ["AI applications in healthcare", "cancer diagnosis"]
    exclusion_criteria = ["agriculture", "robotics outside healthcare"]

    try:
        included = agent.predict_inclusion(papers, inclusion_criteria, exclusion_criteria)
        print("✅ Included papers:")
        for paper in included:
            print(f"- {paper['title']}")
    except Exception as e:
        print(f"❌ Error during screening: {e}")

if __name__ == "__main__":
    test_abstract_screening_agent()
