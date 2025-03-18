import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from joblib import dump

# Load the dataset
telecom_cust = pd.read_csv(r'C:\Users\intel\Documents\OneDrive_1_15-02-2025-20250223T092247Z-001\OneDrive_1_15-02-2025\Telco_Customer_Churn.csv')

# Data preprocessing
# Fill missing values in 'TotalCharges' and convert to numeric
telecom_cust['TotalCharges'] = pd.to_numeric(telecom_cust['TotalCharges'], errors='coerce')
telecom_cust['TotalCharges'] = telecom_cust['TotalCharges'].fillna(0)

# Convert 'Churn' to binary labels
label_encoder_churn = LabelEncoder()
telecom_cust['Churn'] = label_encoder_churn.fit_transform(telecom_cust['Churn'])

# Use Label Encoding for 'InternetService' and 'Contract'
label_encoder_is = LabelEncoder()
label_encoder_c = LabelEncoder()
telecom_cust['InternetService'] = label_encoder_is.fit_transform(telecom_cust['InternetService'])
telecom_cust['Contract'] = label_encoder_c.fit_transform(telecom_cust['Contract'])

# Select features
selected_features = ['tenure', 'InternetService', 'Contract', 'MonthlyCharges', 'TotalCharges']
X = telecom_cust[selected_features]
y = telecom_cust['Churn']

# Scale the features for SVM
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train the SVM model
model = SVC(random_state=101,class_weight='balanced')  # Enable probability estimation for debugging
model.fit(X_scaled, y)

# Save the trained model and scaler to a file
dump(model, 'svm_model.joblib')
dump(scaler, 'scaler.joblib')

print(telecom_cust['Churn'].value_counts())
