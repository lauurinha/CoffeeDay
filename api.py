# api.py
from flask import Flask, request, jsonify, send_from_directory
import numpy as np
from model import simple_nn

app = Flask(__name__)

# Route: serve the frontend HTML
@app.route("/")
def home():
    return send_from_directory(".", "index.html")

# Route: prediction endpoint
@app.route("/predict", methods=["POST"])
def predict():
    # Expect JSON data from frontend
    data = request.get_json()

    # Extract values safely (default = 0 if not provided)
    x1 = float(data.get("x1", 0))
    x2 = float(data.get("x2", 0))

    # Run through model
    result = simple_nn(np.array([x1, x2]))

    # Send back JSON response
    return jsonify({"prediction": float(result)})

if __name__ == "__main__":
    app.run(debug=True)
