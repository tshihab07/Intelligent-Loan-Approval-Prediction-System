# app/routes.py
from flask import Blueprint, render_template, request, redirect, url_for, flash, jsonify
from flask_login import login_user, logout_user, login_required, current_user
from app.models import predict_loan_approval, LoanApplication, User
from app.forms import UserInfoForm, LoanApplicationForm, AdminLoginForm
from app import db
import random

bp = Blueprint('main', __name__)

@bp.route('/')
def index():
    """Home page - first page users see"""
    return render_template('index.html')

# Store user data in session or pass via POST
user_data_store = {}

@bp.route('/basic-information', methods=['GET', 'POST'])
def basic_info():
    """Basic information form"""
    form = UserInfoForm()
    if form.validate_on_submit():
        # Store user data temporarily
        user_data_store['basic_info'] = {
            'full_name': form.full_name.data,
            'mobile_number': form.mobile_number.data,
            'address': form.address.data,
            'street': form.street.data,
            'zip_code': form.zip_code.data,
            'current_city': form.current_city.data,
            'email': form.email.data,
            'nid': form.nid.data
        }
        return redirect(url_for('main.apply'))
    return render_template('basic-information.html', form=form)

@bp.route('/apply', methods=['GET', 'POST'])
def apply():
    form = LoanApplicationForm()
    if form.validate_on_submit():
        # Get user data from store
        basic_info = user_data_store.get('basic_info', {})
        
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
        
        # Generate unique 6-digit loan ID
        while True:
            loan_id = random.randint(100000, 999999)
            if not LoanApplication.query.filter_by(loan_id=loan_id).first():
                break
        
        # Save actual user data
        loan_app = LoanApplication(
            loan_id=loan_id,
            full_name=basic_info.get('full_name', 'User'),
            mobile_number=basic_info.get('mobile_number', '1234567890'),
            address=basic_info.get('address', 'Address'),
            street=basic_info.get('street', ''),
            zip_code=basic_info.get('zip_code', '12345'),
            current_city=basic_info.get('current_city', 'City'),
            email=basic_info.get('email', ''),
            nid=basic_info.get('nid', '123456789'),
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
                               loan_id=loan_id)
    return render_template('apply.html', form=form)

@bp.route('/check-loan-status')
def check_loan_status():
    loan_id = request.args.get('loan_id')
    if loan_id and loan_id.isdigit() and len(loan_id) == 6:
        application = LoanApplication.query.filter_by(loan_id=int(loan_id)).first()
        if application:
            return redirect(url_for('main.loan_status', loan_id=application.loan_id))
        else:
            flash(f"No loan application found with ID {loan_id}", "warning")
    else:
        flash("Please enter a valid 6-digit loan ID", "warning")
    return redirect(url_for('main.index'))

# 👇 NEW: Detailed loan status page
@bp.route('/loan-status/<int:loan_id>')
def loan_status(loan_id):
    application = LoanApplication.query.filter_by(loan_id=loan_id).first_or_404()
    
    # Calculate remaining duration for re-apply (example: 30 days)
    from datetime import datetime, timedelta
    today = datetime.now()
    reapply_date = application.timestamp + timedelta(days=30)
    days_remaining = (reapply_date - today).days
    
    return render_template('loan_status.html', 
                           application=application,
                           days_remaining=max(0, days_remaining))

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