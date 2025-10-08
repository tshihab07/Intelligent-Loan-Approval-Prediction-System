# app/routes.py
from flask import Blueprint, render_template, request, redirect, url_for, flash, jsonify
from flask_login import login_user, logout_user, login_required, current_user
from app.models import predict_loan_approval, LoanApplication, User
from app.forms import UserInfoForm, LoanApplicationForm, AdminLoginForm
from app import db

bp = Blueprint('main', __name__)

@bp.route('/')
def index():
    form = UserInfoForm()
    return render_template('index.html', form=form)

@bp.route('/apply', methods=['GET', 'POST'])
def apply():
    form = LoanApplicationForm()
    if form.validate_on_submit():
        # Store user info in session or pass through (simplified: redirect to result with POST)
        user_data = {
            'gender': form.gender.data,
            'marital_status': form.marital_status.data,
            'dependents': form.dependents.data,
            'education': form.education.data,
            'self_employed': form.self_employed.data,
            'applicant_income': form.applicant_income.data,
            'co_applicant_income': form.co_applicant_income.data or 0,
            'loan_amount': form.loan_amount.data,
            'loan_term': form.loan_term.data,
            'credit_history': form.credit_history.data,
            'property_area': form.property_area.data
        }
        
        prediction, proba, features = predict_loan_approval(user_data)
        
        # Create loan application record (user info will be added in real implementation)
        loan_app = LoanApplication(
            full_name="User",  # In real app, get from session
            mobile_number="1234567890",
            address="Address",
            zip_code="12345",
            current_city="City",
            nid="123456789",
            gender=user_data['gender'],
            marital_status=user_data['marital_status'],
            dependents=user_data['dependents'],
            education=user_data['education'],
            self_employed=user_data['self_employed'],
            applicant_income=user_data['applicant_income'],
            co_applicant_income=user_data['co_applicant_income'],
            loan_amount=user_data['loan_amount'],
            loan_term=user_data['loan_term'],
            credit_history=user_data['credit_history'],
            property_area=user_data['property_area'],
            prediction=int(prediction),
            probability=float(proba)
        )
        db.session.add(loan_app)
        db.session.commit()
        
        return render_template('result.html', 
                               status=loan_app.status,
                               color=loan_app.status_color,
                               proba_percent=round(proba * 100, 2),
                               application_id=loan_app.id)
    return render_template('apply.html', form=form)

@bp.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('main.admin_dashboard'))
    form = AdminLoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user and user.check_password(form.password.data):
            login_user(user)
            return redirect(url_for('main.admin_dashboard'))
        flash('Invalid username or password', 'error')
    return render_template('admin/login.html', form=form)

@bp.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('main.index'))

@bp.route('/admin')
@login_required
def admin_dashboard():
    page = request.args.get('page', 1, type=int)
    status_filter = request.args.get('status', 'all')
    
    query = LoanApplication.query
    if status_filter == 'approved':
        query = query.filter_by(prediction=1)
    elif status_filter == 'rejected':
        query = query.filter_by(prediction=0)
    
    applications = query.order_by(LoanApplication.timestamp.desc()).paginate(
        page=page, per_page=10, error_out=False
    )
    
    return render_template('admin/dashboard.html', applications=applications, status_filter=status_filter)

@bp.route('/admin/export')
@login_required
def export_data():
    applications = LoanApplication.query.all()
    data = []
    for app in applications:
        data.append({
            'ID': app.id,
            'Full Name': app.full_name,
            'Mobile': app.mobile_number,
            'Loan Amount': app.loan_amount,
            'Status': app.status,
            'Probability': app.probability,
            'Timestamp': app.timestamp
        })
    return jsonify(data)