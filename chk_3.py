
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import streamlit as st

# Load the trained model and label encoders
model = joblib.load('model.pkl')
label_encoder = joblib.load('encoder.pkl')

# Function to encode user inputs
def encode_inputs(input_data, encoders):
    for column, encoder in encoders.items():
        if input_data[column].iloc[0] not in encoder.classes_:
            # Handle unseen labels by assigning a default value or raise an exception
            input_data[column] = encoder.transform([encoder.classes_[0]])[0]  # Default to the first class
        else:
            input_data[column] = encoder.transform([input_data[column].iloc[0]])[0]
    return input_data

# Streamlit interface
st.title("Bank Account Prediction")

Country = st.text_input("country")
Uniqueid = st.text_input("unique ID")
# Bank_account = st.text_input("Bank Account")
Location_type = st.text_input("location Type")
Cellphone_access = st.text_input("cellphone Access")
Gender_of_respondent = st.text_input("gender of Respondent")
Relationship_with_head = st.text_input("relationship with Head")
Marital_status = st.text_input("marital Status")
Education_level = st.text_input("education Level")
Job_type = st.text_input("job Type")
Year = st.number_input("year")
Household_size = st.number_input("household Size")
Age_of_respondent = st.number_input("age of Respondent")

# Create DataFrame from user input
df = pd.DataFrame({
    "country": [Country],
    "year": [Year],
    "uniqueid": [Uniqueid],
    # "bank_account": [Bank_account],
    "location_type": [Location_type],
    "cellphone_access": [Cellphone_access],
    "household_size": [Household_size],
    "age_of_respondent": [Age_of_respondent],
    "gender_of_respondent": [Gender_of_respondent],
    
    
    "relationship_with_head": [Relationship_with_head],
    "marital_status": [Marital_status],
    "education_level": [Education_level],
    "job_type": [Job_type]
    
    
    
})
# Index(['country', 'year', 'uniqueid', 'location_type', 'cellphone_access',
#        'household_size', 'age_of_respondent', 'gender_of_respondent',
#        'relationship_with_head', 'marital_status', 'education_level',
#        'job_type'],
#       dtype='object')
 
#  Index(['country', 'year', 'uniqueid', 'bank_account', 'location_type',
#        'cellphone_access', 'household_size', 'age_of_respondent',
#        'gender_of_respondent', 'relationship_with_head', 'marital_status',
#        'education_level', 'job_type'],
# df1=df.drop(['year', 'household_size', 'age_of_respondent'],axis=1) 
# Encode the inputs

df = encode_inputs(df, label_encoder)
# print(df.columns)
# Predict button
if st.button('Predict'):
    # Convert input data to DataFrame
    input_df = pd.DataFrame(df)

    # Ensure all features are present
    input_df = input_df.reindex(columns=df.columns, fill_value=0)

    # Make prediction
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)

    # Display the prediction
    st.write(f'The prediction is: {"Bank Account" if prediction[0] else "No Bank Account"}')
    st.write(f'Prediction probabilities: {prediction_proba[0]}')
