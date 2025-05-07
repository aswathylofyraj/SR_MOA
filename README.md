# Systematic Review Screening with LLM Agents

This project automates systematic literature reviews using a pipeline of intelligent agents. It uses powerful LLM-based tools to search, screen, and review academic papers based on user-defined criteria.

---

## 🔍 Overview

The pipeline processes user input (title, inclusion/exclusion criteria, and minimum year) and automatically retrieves and filters papers from arXiv and Semantic Scholar. It performs:

- **Search** using query terms
- **Time filtering**
- **Title & abstract screening** using DeepSeek
- **Quality assessment** of selected papers
- **Review summarization**

All of this is done through a seamless agent-based workflow, producing selected papers, scores, and a summarized review.

---

## 🧠 Workflow

![Workflow](workflow.jpg)

---

## ⚙️ Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/anoopkdcs/SystematicReviewScreening.git

# 1. Clone the repo
git clone https://github.com/anoopkdcs/SystematicReviewScreening.git

# 2. Go into the project folder
cd SystematicReviewScreening/llm-litreview-agents

# 3. (Optional but good) Create and activate a virtual environment
python3 -m venv venv

# 4. Activate the virtual environment


# If you're on Windows CMD:
venv\Scripts\activate

# If you're on Windows PowerShell:
.\venv\Scripts\Activate.ps1

# 5. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 6. Run the app (assuming app file is app.py and using uvicorn)
uvicorn main:app --reload

📦 Output
The system returns:

Filtered and scored research papers

AI-generated summary review