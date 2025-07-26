# 🚀 Customer Churn Prediction API with FastAPI
This project provides a machine learning model with a REST API using FastAPI to predict whether a customer will churn (i.e., stop using a subscription service). It's built using historical customer data such as age, balance, credit score, and behavior indicators.

## 📂 Project Structure
```bash
customer-churn-api/
│
├── main.py                     
├── model/
│   └── churn_model.pkl        
├── requirements.txt            
└── README.md
```

## 🧠Model
The API uses a pre-trained machine learning model (e.g., Logistic Regression, Random Forest, or XGBoost) trained on historical customer data. The model should be saved as a .pkl file or similar.

## 📥 Installation
### Clone the repository
```bash
git clone https://github.com/yourusername/customer-churn-api.git
cd customer-churn-api
```
### (Optional) Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
### Install dependencies
```bash
pip install -r requirements.txt
```
## ▶️ Running the API
```bash
uvicorn main:app --reload
```

## 📦 requirements.txt
```bash
fastapi
uvicorn
scikit-learn
pandas
joblib
```

## 📤 Example Request
Endpoint
```bash
POST /predict
```
Sample Payload
```bash
{
  "gender": "Female",
  "senior_citizen": 0,
  "partner": "Yes",
  "dependents": "No",
  "tenure": 12,
  "phone_service": "Yes",
  "multiple_lines": "No",
  "internet_service": "Fiber optic",
  "online_security": "No",
  "online_backup": "Yes",
  "device_protection": "No",
  "tech_support": "No",
  "streaming_tv": "Yes",
  "streaming_movies": "Yes",
  "contract": "Month-to-month",
  "paperless_billing": "Yes",
  "payment_method": "Electronic check",
  "monthly_charges": 85.5,
  "total_charges": 1020.75
}
```
Sample Response
```bash
{
  "churn_probability": 0.82,
  "prediction": "Churn"
}
```
