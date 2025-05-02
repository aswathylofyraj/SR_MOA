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

