from flask import Flask, request, render_template, jsonify
from dotenv import load_dotenv
import os
import requests

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Hugging Face API setup
API_URL = "https://api-inference.huggingface.co/models/gpt2"
API_KEY = os.getenv("HF_API_KEY")

headers = {"Authorization": f"Bearer {API_KEY}"}

def query_huggingface(payload):
    """Send a query to Hugging Face API."""
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        # Adjust to handle JSON input from the frontend
        data = request.get_json()  # Get the JSON from the request body
        user_input = data.get("message")
        
        # Prepare payload for Hugging Face API
        payload = {"inputs": user_input}
        response = query_huggingface(payload)
        
        # Extract the response text
        chatbot_reply = response.get("generated_text", "Sorry, I didn't understand that.")
        
        # Return the response as JSON
        return jsonify({"reply": chatbot_reply})

    return render_template("index.html")

if __name__ == "__main__":
    # Make Flask listen on the correct port provided by Render
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 10000)))