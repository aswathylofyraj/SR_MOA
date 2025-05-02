import httpx

class AbstractScreeningAgent:
    def __init__(self):
        self.api_key = "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"  # 🔐 Direct API key
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

    def predict_inclusion(self, papers, inclusion_criteria, exclusion_criteria):
        abstracts = [paper.get("abstract", "") for paper in papers]
        prompt = self._generate_prompt(abstracts, inclusion_criteria, exclusion_criteria)

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
        abstract_idx = 0

        for line in lines:
            if line.strip().startswith(str(abstract_idx + 1) + "."):
                if "YES" in line.upper():
                    included_papers.append(papers[abstract_idx])
                abstract_idx += 1

        if not included_papers and any("YES" in l.upper() for l in lines):
            included_papers = [paper for i, paper in enumerate(papers) if "YES" in lines[i].upper()]

        return included_papers
