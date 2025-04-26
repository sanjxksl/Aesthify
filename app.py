from flask import Flask, render_template
from routes import routes  # Blueprint containing all routes

from utils import *

# Initialize the Flask web application & Fixing Image Input Length
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH']
app.register_blueprint(routes)  # Attach all routes

@app.route('/')
def index():
    """
    Endpoint: / [GET]

    Serves the main frontend page — index.html — which is expected to provide
    the interface for image upload and UI for invoking aesthetic evaluation.
    """
    return render_template('index.html')

if __name__ == "__main__":
    """
    Launches the Flask development server.

    Note: Not for production use. Use Gunicorn or similar WSGI server for deployment.
    """
    app.run(debug=True)
