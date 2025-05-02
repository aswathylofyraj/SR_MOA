import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
import joblib  # Import joblib for saving the model

def load_data(data_file):
    # Load the dataset
    data = pd.read_csv(data_file)
    return data

def train_abstract_screening_agent(data_file):
    # Load the data
    data = load_data(data_file)
    
    # Split the data into features and labels
    X = data['Abstract']
    y = data['Label']
    
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create a pipeline that combines CountVectorizer and MultinomialNB
    model = make_pipeline(CountVectorizer(), MultinomialNB())
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy:.2f}")
    
    # Save the model
    joblib.dump(model, 'abstract_screening_model.pkl')
    print("Abstract screening model saved as 'abstract_screening_model.pkl'")

def main():
    data_file = 'backend/data/combined_data.csv'  # Path to your cleaned data
    train_abstract_screening_agent(data_file)

if __name__ == "__main__":
    main()