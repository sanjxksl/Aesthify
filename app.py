"""
===============================================================================
Aesthify Web Application Entry Point
===============================================================================
Initializes the Flask app, configures application parameters, registers routes, 
and defines the main route for serving the frontend UI.

Note:
- For production deployment, use Gunicorn or a WSGI-compliant server.
"""

from flask import Flask, render_template
from routes import routes  # Import application routes (Blueprint)
from utils import *         # Import utility functions

# Initialize Flask web application
app = Flask(__name__)

# Set maximum allowed payload for file uploads (To be configured appropriately)
app.config['MAX_CONTENT_LENGTH']

# Register routes from the routes Blueprint
app.register_blueprint(routes)

@app.route('/')
def index():
    """
    Endpoint: /
    Method: GET

    Renders the main frontend page 'index.html'.
    Purpose:
    - Provides interface for users to upload images.
    - Enables interaction with aesthetic evaluation features.
    """
    return render_template('index.html')

if __name__ == "__main__":
    """
    Development Server Launch

    Runs the Flask application in debug mode for local development and testing.
    Note:
    - Debug mode auto-reloads on code changes.
    - Do NOT use this server in production environments.
    """
    app.run(debug=True)