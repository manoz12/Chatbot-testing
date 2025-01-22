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
        user_input = request.form.get("message")
        payload = {"inputs": user_input}
        response = query_huggingface(payload)
        chatbot_reply = response.get("generated_text", "Sorry, I didn't understand that.")
        return jsonify({"reply": chatbot_reply})

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
