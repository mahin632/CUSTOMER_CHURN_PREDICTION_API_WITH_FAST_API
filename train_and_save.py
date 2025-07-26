

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
import joblib

# Load data
df = pd.read_csv("churn_data.csv.csv")
df = df.drop(columns=['RowNumber', 'CustomerId', 'Surname'])

# Encode categorical
for col in ['Geography', 'Gender']:
    df[col] = LabelEncoder().fit_transform(df[col])

X = df.drop('Exited', axis=1)
y = df['Exited']

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, stratify=y, random_state=42)
model = GradientBoostingClassifier().fit(X_train, y_train)

# Save model and scaler
joblib.dump(model, 'model.pkl')
joblib.dump(scaler, 'scaler.pkl')
