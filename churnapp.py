from joblib import load
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import streamlit as st

# Load the trained model and scaler
model = load('svm_model.joblib')
scaler = load('scaler.joblib')

# Load the dataset to recreate encoders (you can reuse the ones from training)
telecom_cust = pd.read_csv(r'C:\Users\intel\Documents\OneDrive_1_15-02-2025-20250223T092247Z-001\OneDrive_1_15-02-2025\Telco_Customer_Churn.csv')


# Streamlit UI
st.title("Customer Churn Prediction App")
st.header("Enter Customer Information")

tenure = st.number_input("Tenure (in months)", min_value=0, max_value=100, value=1)
internet_service = st.selectbox("Internet Service", telecom_cust['InternetService'].unique())
contract = st.selectbox("Contract", telecom_cust['Contract'].unique())
monthly_charges = st.number_input("Monthly Charges", min_value=0, max_value=200, value=50)
total_charges = st.number_input("Total Charges", min_value=0, max_value=10000, value=0)

# Apply the same LabelEncoder as in training

# Initialize the encoder
label_encoder_is = LabelEncoder()
internet_service = label_encoder_is.transform([internet_service])[0]
contract = label_encoder_c.transform([contract])[0]

# Prepare the input data (scaled the same way as training data)
input_data = [[tenure, internet_service, contract, monthly_charges, total_charges]]
input_scaled = scaler.transform(input_data)

# Make a prediction
prediction = model.predict(input_scaled)

# Display the result
st.header("Prediction Result")
if prediction[0] == 0:
    st.success("This customer is likely to stay.")
else:
    st.error("This customer is likely to churn.")

# Debugging: Show prediction probability (if SVC is used with probability=True)
#prediction_proba = model.predict_proba(input_scaled)
#st.write(f"Prediction probabilities: {prediction_proba}")
