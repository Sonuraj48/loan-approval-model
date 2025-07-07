import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# Load dataset
df = pd.read_csv('loan_data.csv')

# Drop missing values for simplicity
df = df.dropna()

# Encode categorical features
cols = ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area']
df[cols] = df[cols].apply(LabelEncoder().fit_transform)

# Features and target
X = df[['Gender', 'Married', 'Education', 'ApplicantIncome', 'LoanAmount', 'Credit_History', 'Property_Area']]
y = df['Loan_Status'].map({'Y': 1, 'N': 0})  # Encode target

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model
joblib.dump(model, 'model.pkl')
print("Model trained and saved.")
