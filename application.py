from flask import Flask, request, render_template, jsonify
from dotenv import load_dotenv
import os
import requests
import time

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Hugging Face API setup
API_URL = "https://api-inference.huggingface.co/models/tiiuae/falcon-7b-instruct"
API_KEY = os.getenv("HF_API_KEY")

headers = {"Authorization": f"Bearer {API_KEY}"}

def query_huggingface(payload):
    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error querying Hugging Face API: {e}")
        time.sleep(5)  # Retry after 5 seconds
        return {"generated_text": "Error occurred, retrying later."}

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        try:
            data = request.get_json()  # Parse the JSON payload
            user_input = data.get("message")
            if not user_input:
                return jsonify({"reply": "No input received. Please type a message."})

            # Query Hugging Face API
            payload = {"inputs": user_input}
            response = query_huggingface(payload)

            # Extract chatbot reply
            chatbot_reply = response.get("generated_text", "Sorry, I didn't understand that.")
            return jsonify({"reply": chatbot_reply})

        except Exception as e:
            print(f"Error: {e}")
            return jsonify({"reply": "An error occurred. Please try again later."})

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 10000)))
