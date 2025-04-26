"""
routes/__init__.py

Initializes the Flask Blueprint for all routes and imports individual route modules.
"""

from flask import Blueprint

# Initialize a single Blueprint object for all routes
routes = Blueprint('routes', __name__)

# Import route handlers (this auto-registers them to `routes`)
from routes.evaluation import evaluate