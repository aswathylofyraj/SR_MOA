import streamlit as st
import joblib

# Load the trained models
title_model = joblib.load('title_screening_model.pkl')
abstract_model = joblib.load('abstract_screening_model.pkl')

def predict_inclusion(title, abstract, inclusion_criteria, exclusion_criteria):
    # Prepare the input for the title model
    title_prediction = title_model.predict([title])[0]
    
    # Prepare the input for the abstract model
    abstract_prediction = abstract_model.predict([abstract])[0]
    
    # Logic to determine inclusion based on predictions
    if title_prediction == 1 and abstract_prediction == 1:
        return "Include"
    else:
        return "Exclude"

def main():
    st.title("Paper Inclusion/Exclusion Predictor")

    # Input fields
    title = st.text_input("Title of the Paper")
    inclusion_criteria = st.text_area("Inclusion Criteria")
    exclusion_criteria = st.text_area("Exclusion Criteria")
    paper = st.file_uploader("Upload the Paper (PDF)", type=["pdf"])

    if st.button("Predict"):
        if paper is not None:
            # Extract title and abstract from the uploaded paper
            # Here you would call your extraction function (currently omitted)
            # For demonstration, we will use dummy values for abstract
            abstract = "This is a dummy abstract for demonstration purposes."  # Replace with actual extraction logic
            
            # Call the prediction function
            result = predict_inclusion(title, abstract, inclusion_criteria, exclusion_criteria)
            st.success(f"The paper should be: {result}")
        else:
            st.error("Please upload a paper.")

if __name__ == "__main__":
    main()