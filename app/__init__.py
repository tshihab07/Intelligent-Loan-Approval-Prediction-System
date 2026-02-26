from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager
from dotenv import load_dotenv
import os


# Load environment variables
load_dotenv()

db = SQLAlchemy()
login_manager = LoginManager()

def create_app():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    app = Flask(
        __name__,
        instance_relative_config=True,
        template_folder=os.path.join(project_root, 'templates'),
        static_folder=os.path.join(project_root, 'static')
    )
    
    app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY',)    
    app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL')
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

    if not app.config['SECRET_KEY']:
        raise ValueError("SECRET_KEY is not set in environment variables")

    if not app.config['SQLALCHEMY_DATABASE_URI']:
        raise ValueError("DATABASE_URL is not set in environment variables")
    
    db.init_app(app)
    login_manager.init_app(app)
    login_manager.login_view = 'main.login'

    from app.routes import bp as main_bp
    from app.models import User
    
    app.register_blueprint(main_bp)
    
    with app.app_context():
        db.create_all()

        # Secure admin creation
        admin_username = os.environ.get('ADMIN_USERNAME')
        admin_password = os.environ.get('ADMIN_PASSWORD')

        if not admin_username or not admin_password:
            raise ValueError("Admin credentials not set in environment variables")
        
        # Create admin user only if it doesn't exist
        admin_user = User.query.filter_by(username=admin_username).first()
        if not admin_user:
            admin = User(username=admin_username)
            admin.set_password(admin_password)
            db.session.add(admin)
            db.session.commit()
            print("Admin user created successfully!")
        
        else:
            print("Admin user already exists.")
    
    return app