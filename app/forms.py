from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, RadioField, IntegerField, PasswordField, validators
from wtforms.validators import DataRequired, Email, NumberRange, Length

class UserInfoForm(FlaskForm):
    # These 4 fields are REQUIRED
    full_name = StringField('Full Name', validators=[DataRequired(message="Full Name is required")])
    mobile_number = StringField('Mobile Number', validators=[DataRequired(message="Mobile Number is required")])
    current_city = StringField('Current City', validators=[DataRequired(message="Current City is required")])
    nid = StringField('NID', validators=[DataRequired(message="NID is required")])
    
    # Optional fields (no email validation)
    address = StringField('Address', validators=[DataRequired()])
    street = StringField('Street (Optional)')
    zip_code = StringField('Zip Code', validators=[DataRequired()])
    email = StringField('Email (Optional)')
    submit = SubmitField('Continue to Loan Application')

class LoanApplicationForm(FlaskForm):
    gender = RadioField('Gender', choices=[('Male', 'Male'), ('Female', 'Female')], validators=[DataRequired()])
    marital_status = RadioField('Marital Status', choices=[('Married', 'Married'), ('Not Married', 'Not Married')], validators=[DataRequired()])
    dependents = RadioField('Dependents', choices=[('0', '0'), ('1', '1'), ('2', '2'), ('3+', '3+')],
                            validators=[DataRequired()])
    education = RadioField('Education', choices=[('Graduate', 'Graduate'), ('Not Graduate', 'Not Graduate')], validators=[DataRequired()])
    self_employed = RadioField('Self-Employed', choices=[('Yes', 'Yes'), ('No', 'No')], validators=[DataRequired()])
    applicant_income = IntegerField('Applicant Income (USD)', validators=[DataRequired(), NumberRange(min=0)])
    co_applicant_income = IntegerField('Co-Applicant Income (USD) (Optional)', validators=[NumberRange(min=0)])
    loan_amount = IntegerField('Loan Amount (USD)', validators=[DataRequired(), NumberRange(min=1)])
    loan_term = RadioField('Loan Term (Years)', choices=[('10', '10'), ('15', '15'), ('20', '20'), ('25', '25'), ('30', '30')],
                           validators=[DataRequired()])
    credit_history = RadioField('Previous Loan Status (Credit History)', choices=[('Yes', 'Yes'), ('No', 'No')], validators=[DataRequired()])
    property_area = RadioField('Property Area', choices=[('Urban', 'Urban'), ('Semi Urban', 'Semi Urban'), ('Rural', 'Rural')],
                               validators=[DataRequired()])
    submit = SubmitField('Submit Loan Application')

class AdminLoginForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired(), Length(min=4, max=20)])
    password = PasswordField('Password', validators=[DataRequired(), Length(min=6)])
    submit = SubmitField('Login')