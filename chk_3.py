import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import streamlit as st

# Load the dataset
data = pd.read_csv('Financial_inclusion_dataset.csv')
data_copy = data.copy()

# Display data info
print(data_copy.head())
print("Data Info")
print(data_copy.info())

# Handling missing and corrupted values
categorical_col = data_copy.select_dtypes(include='object').columns
numerical_col = data_copy.select_dtypes(include='number').columns

data_copy[categorical_col] = data_copy[categorical_col].fillna(data_copy[categorical_col].mode().iloc[0])
data_copy[numerical_col] = data_copy[numerical_col].fillna(data_copy[numerical_col].mean())



# Identify and remove outliers
def identify_outliers(column):
    Q1 = column.quantile(0.25)
    Q3 = column.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return lower_bound, upper_bound

def remove_outliers(data_copy):
    for col in data_copy.select_dtypes(include=['number']).columns:
        lower_bound, upper_bound = identify_outliers(data_copy[col])
        data_copy = data_copy[(data_copy[col] >= lower_bound) & (data_copy[col] <= upper_bound)]
    return data_copy

data_copy = remove_outliers(data_copy)

# Encode categorical features using LabelEncoder
label_encoders = {}
for column in data_copy.select_dtypes(include=['object']).columns:
    label_encoders[column] = LabelEncoder()
    data_copy[column] = label_encoders[column].fit_transform(data_copy[column])

# Save the label encoders
joblib.dump(label_encoders, 'label_encoders.pkl')
# Machine Learning
X = data_copy.drop("bank_account", axis=1)
y = data_copy["bank_account"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save the model to a .pkl file
joblib.dump(model, 'model.pkl')

# Make predictions and evaluate the model
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)

# Save the cleaned data to an Excel file
data_copy.to_excel('cleaned_data.xlsx', index=False)

# Load the trained model and label encoders
model = joblib.load('model.pkl')
label_encoders = joblib.load('label_encoders.pkl')

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

Country = st.text_input("Country")
Uniqueid = st.text_input("Unique ID")
Bank_account = st.text_input("Bank Account")
Location_type = st.text_input("Location Type")
Cellphone_access = st.text_input("Cellphone Access")
Gender_of_respondent = st.text_input("Gender of Respondent")
Relationship_with_head = st.text_input("Relationship with Head")
Marital_status = st.text_input("Marital Status")
Education_level = st.text_input("Education Level")
Job_type = st.text_input("Job Type")
Year = st.number_input("Year")
Household_size = st.number_input("Household Size")
Age_of_respondent = st.number_input("Age of Respondent")

# Create DataFrame from user input
df = pd.DataFrame({
    "country": [Country],
    "uniqueid": [Uniqueid],
    "bank_account": [Bank_account],
    "location_type": [Location_type],
    "cellphone_access": [Cellphone_access],
    "gender_of_respondent": [Gender_of_respondent],
    "relationship_with_head": [Relationship_with_head],
    "marital_status": [Marital_status],
    "education_level": [Education_level],
    "job_type": [Job_type],
    "year": [Year],
    "household_size": [Household_size],
    "age_of_respondent": [Age_of_respondent]
})

# Encode the inputs
df = encode_inputs(df, label_encoders)

# Predict button
if st.button('Predict'):
    # Convert input data to DataFrame
    input_df = pd.DataFrame(df)

    # Ensure all features are present
    input_df = input_df.reindex(columns=X.columns, fill_value=0)

    # Make prediction
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)

    # Display the prediction
    st.write(f'The prediction is: {"Bank Account" if prediction[0] else "No Bank Account"}')
    st.write(f'Prediction probabilities: {prediction_proba[0]}')
