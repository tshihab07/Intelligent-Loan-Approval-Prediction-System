import joblib
import pandas as pd
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import UserMixin
from app import db, login_manager

model = joblib.load("artifacts/models/xgboost.pkl")

FEATURES = [
    'Dependents', 'LoanAmount', 'CreditHistory', 'TotalIncome',
    'IncomeLoanRatio', 'LoanTermYears', 'Has_CoApplicantIncome',
    'Gender_Male', 'Married_Yes', 'Education_NotGraduate',
    'PropertyArea_Semiurban', 'PropertyArea_Urban', 'SelfEmployed_Yes'
]

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)
    
    def set_password(self, password):
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

class LoanApplication(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    loan_id = db.Column(db.Integer, unique=True, nullable=False)  # 6-digit unique ID
    full_name = db.Column(db.String(100), nullable=False)
    mobile_number = db.Column(db.String(20), nullable=False)
    address = db.Column(db.String(200), nullable=False)
    street = db.Column(db.String(100))
    zip_code = db.Column(db.String(10), nullable=False)
    current_city = db.Column(db.String(50), nullable=False)
    email = db.Column(db.String(100))
    nid = db.Column(db.String(20), nullable=False)
    gender = db.Column(db.String(10), nullable=False)
    marital_status = db.Column(db.String(20), nullable=False)
    dependents = db.Column(db.String(10), nullable=False)
    education = db.Column(db.String(20), nullable=False)
    self_employed = db.Column(db.String(10), nullable=False)
    applicant_income = db.Column(db.Float, nullable=False)
    co_applicant_income = db.Column(db.Float, default=0.0)
    loan_amount = db.Column(db.Float, nullable=False)
    loan_term = db.Column(db.Integer, nullable=False)
    credit_history = db.Column(db.String(10), nullable=False)
    property_area = db.Column(db.String(20), nullable=False)
    prediction = db.Column(db.Integer, nullable=False)
    probability = db.Column(db.Float, nullable=False)
    timestamp = db.Column(db.DateTime, default=db.func.current_timestamp())
    
    @property
    def status(self):
        return "Approved" if self.prediction == 1 else "Rejected"
    
    @property
    def status_color(self):
        return "success" if self.prediction == 1 else "danger"

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

def predict_loan_approval(user_data):
    applicant_income = float(user_data['applicant_income'])
    co_applicant_income = float(user_data['co_applicant_income']) if user_data['co_applicant_income'] else 0.0
    total_income = applicant_income + co_applicant_income
    loan_amount = float(user_data['loan_amount'])
    income_loan_ratio = total_income / loan_amount if loan_amount > 0 else 0.0
    has_co_applicant_income = 1 if co_applicant_income > 0 else 0
    
    features = {
        'Dependents': int(user_data['dependents'].replace('+', '')) if user_data['dependents'] != '0' else 0,
        'LoanAmount': loan_amount,
        'CreditHistory': 1 if user_data['credit_history'] == 'Yes' else 0,
        'TotalIncome': total_income,
        'IncomeLoanRatio': income_loan_ratio,
        'LoanTermYears': int(user_data['loan_term']),
        'Has_CoApplicantIncome': has_co_applicant_income,
        'Gender_Male': 1 if user_data['gender'] == 'Male' else 0,
        'Married_Yes': 1 if user_data['marital_status'] == 'Married' else 0,
        'Education_NotGraduate': 1 if user_data['education'] == 'Not Graduate' else 0,
        'PropertyArea_Semiurban': 1 if user_data['property_area'] == 'Semi Urban' else 0,
        'PropertyArea_Urban': 1 if user_data['property_area'] == 'Urban' else 0,
        'SelfEmployed_Yes': 1 if user_data['self_employed'] == 'Yes' else 0
    }
    
    df = pd.DataFrame([features])
    df = df[FEATURES]
    
    proba = model.predict_proba(df)[0][1]
    prediction = model.predict(df)[0]
    
    return prediction, proba, features