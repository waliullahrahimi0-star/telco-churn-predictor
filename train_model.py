import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load and clean data
df = pd.read_csv('https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv')
df = df.drop('customerID', axis=1)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df = df.dropna()
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# Prepare features
X = pd.get_dummies(df.drop('Churn', axis=1), drop_first=True)
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Train and save
model = RandomForestClassifier(class_weight='balanced', n_estimators=200, random_state=42)
model.fit(X_train, y_train)

joblib.dump(model, 'best_churn_model.pkl')
joblib.dump(X.columns.tolist(), 'feature_columns.pkl')

print("✅ Model saved! You can now download best_churn_model.pkl and feature_columns.pkl")
