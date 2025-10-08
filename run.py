# run.py
import os
from app import create_app

# Get the directory where run.py is located (project root)
project_root = os.path.dirname(os.path.abspath(__file__))

# Create the app with explicit template and static folders
app = create_app()

if __name__ == '__main__':
    app.run(debug=True)