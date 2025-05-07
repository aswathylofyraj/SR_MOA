import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from agents.title_agent import TitleScreeningAgent
from agents.abstract_agent import AbstractScreeningAgent

# Load and preprocess data
df = pd.read_excel("data/HPV_corpus.xlsx")
df = df.dropna(subset=["Title", "Abstract", "Label"])

# Normalize and map labels to 0/1
df["Label"] = df["Label"].astype(str).str.strip().str.lower().map({
    "include": 1, "included": 1, "yes": 1, "relevant": 1,
    "exclude": 0, "excluded": 0, "no": 0, "irrelevant": 0
})
df = df.dropna(subset=["Label"])
df["Label"] = df["Label"].astype(int)

# Take only the first 5 papers
df = df.iloc[:5]
true_labels = df["Label"].tolist()

# Prepare paper dicts
papers = [
    {
        "title": row["Title"],
        "abstract": row["Abstract"],
        "authors": [],
        "year": "n.d."
    }
    for _, row in df.iterrows()
]

# Criteria
inclusion_criteria = ["HPV"]
exclusion_criteria = ["non-human", "animal study", "not in English"]

# Load agents
title_agent = TitleScreeningAgent()
abstract_agent = AbstractScreeningAgent()

# Title screening
title_screened = title_agent.predict_inclusion(papers, inclusion_criteria, exclusion_criteria)

# Abstract screening
abstract_screened = abstract_agent.predict_inclusion(title_screened, inclusion_criteria, exclusion_criteria)

# Create set of included paper IDs
included_ids = set(id(p) for p in abstract_screened)

# Generate predictions
predicted_labels = [1 if id(p) in included_ids else 0 for p in papers]

# Evaluation
accuracy = accuracy_score(true_labels, predicted_labels)
precision = precision_score(true_labels, predicted_labels, zero_division=0)
recall = recall_score(true_labels, predicted_labels, zero_division=0)
f1 = f1_score(true_labels, predicted_labels, zero_division=0)

# Add predictions to DataFrame
df["Predicted_Label"] = predicted_labels




# Output results
print("\nEvaluation Results (First 5 Papers):")
print(f"Accuracy:  {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall:    {recall:.2f}")
print(f"F1 Score:  {f1:.2f}")
print("Results saved to 'evaluation_results.csv'")
# Save only scores to CSV
scores_df = pd.DataFrame([{
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1 Score": f1
}])
scores_df.to_csv("evaluation_scores.csv", index=False)