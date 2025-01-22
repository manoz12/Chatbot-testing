from flask import Flask, request, render_template, jsonify
from dotenv import load_dotenv
import os
import requests
import time

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Hugging Face API setup with DialoGPT-medium model
API_URL = "https://api-inference.huggingface.co/models/microsoft/DialoGPT-medium"
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
            # Get JSON from the frontend
            data = request.get_json()  # Parse the JSON payload
            print(f"Received data from frontend: {data}")  # Debugging: log received data
            
            # Extract the user input
            user_input = data.get("message")
            print(f"User input: {user_input}")  # Debugging: log user input
            
            # Check if user input is valid
            if not user_input:
                return jsonify({"reply": "No input received. Please type a message."})
            
            # Query Hugging Face API
            payload = {"inputs": user_input}
            response = query_huggingface(payload)
            print(f"Response from Hugging Face API: {response}")  # Debugging: log API response
            
            # Extract chatbot reply
            # DialoGPT returns a list, so we need to access the first item in the list
            chatbot_reply = response[0].get("generated_text", "Sorry, I didn't understand that.")
            return jsonify({"reply": chatbot_reply})
        
        except Exception as e:
            print(f"Error: {e}")  # Log any error for debugging
            return jsonify({"reply": "An error occurred. Please try again later."})

    return render_template("index.html")

if __name__ == "__main__":
    # Make Flask listen on the correct port provided by Render
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 10000)))
