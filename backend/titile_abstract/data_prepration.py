import pandas as pd

def load_data(hpv_file, papd_file):
    # Load the datasets
    hpv_data = pd.read_excel(hpv_file)
    papd_data = pd.read_excel(papd_file)
    return hpv_data, papd_data

def load_criteria(inclusion_file, exclusion_file):
    # Load inclusion and exclusion criteria
    with open(inclusion_file, 'r') as f:
        inclusion_criteria = f.read()
    
    with open(exclusion_file, 'r') as f:
        exclusion_criteria = f.read()
    
    return inclusion_criteria, exclusion_criteria

def preprocess_data(hpv_data, papd_data):
    # Combine datasets
    combined_data = pd.concat([hpv_data, papd_data], ignore_index=True)
    
    # Fill NaN values with an empty string
    combined_data['Title'] = combined_data['Title'].fillna('')
    combined_data['Abstract'] = combined_data['Abstract'].fillna('')
    
    # Drop rows where any column is NaN
    combined_data = combined_data.dropna()
    
    # Convert 'Title' and 'Abstract' to lowercase
    combined_data['Title'] = combined_data['Title'].str.lower()
    combined_data['Abstract'] = combined_data['Abstract'].str.lower()
    
    # Ensure data types are string
    combined_data['Title'] = combined_data['Title'].astype(str)
    combined_data['Abstract'] = combined_data['Abstract'].astype(str)
    
    # Map labels to binary values
    combined_data['Label'] = combined_data['Label'].map({'include': 1, 'exclude': 0})
    
    # Drop rows where 'Label' is NaN after mapping
    combined_data = combined_data.dropna(subset=['Label'])
    
    # Print the shape of the cleaned dataset
    print("Shape of the cleaned dataset:", combined_data.shape)
    
    return combined_data

def main():
    hpv_file = 'backend/data/HPV_corpus.xlsx'
    papd_file = 'backend/data/PAPD_corpus.xlsx'
    inclusion_file = 'backend/titile_abstract/inclusion_criteria.txt'
    exclusion_file = 'backend/titile_abstract/exclusion_criteria.txt'
    
    hpv_data, papd_data = load_data(hpv_file, papd_file)
    inclusion_criteria, exclusion_criteria = load_criteria(inclusion_file, exclusion_file)
    combined_data = preprocess_data(hpv_data, papd_data)
    
    # Save the processed data for further use
    combined_data.to_csv('backend/data/combined_data.csv', index=False)
    print("Data preparation complete.")

if __name__ == "__main__":
    main()