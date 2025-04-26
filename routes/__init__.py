"""
===============================================================================
Aesthify Routes Initialization
===============================================================================
Sets up the Flask Blueprint for the application and imports route handlers.

Blueprint:
- 'routes' — Groups all route modules under a single namespace.
"""

from flask import Blueprint

# Initialize the main Blueprint for all API endpoints
routes = Blueprint('routes', __name__)

# Import route handlers to register them automatically
from routes.evaluation import evaluate