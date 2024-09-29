from flask import Flask, request, jsonify, render_template
from pymongo import MongoClient
from transformers import pipeline
from flask_cors import CORS  # Import CORS

# Initialize Flask application
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# MongoDB connection
client = MongoClient("mongodb://localhost:27017/")  # Adjust the URI as needed
db = client['alumni_db']  # Database name
collection = db['alumni_data']  # Collection name

# Load LLM using a locally available model (ChatGPT-like model)
llm = pipeline("text-generation", model="gpt2")  # This uses GPT-2 as an example

# Home route
@app.route('/')
def home():
    return render_template("index.html")  # Serve the index.html page

# Route to handle chat queries
@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    question = data.get("question", "")

    # Check if the question is valid
    if not question:
        return jsonify({"error": "Please provide a question to answer."}), 400

    # Use LLM to interpret the question
    llm_response = llm(question, max_length=50, num_return_sequences=1)
    response_text = llm_response[0]['generated_text']

    # Example Queries to handle
    if "highest score" in question.lower():
        # Fetch the student with the highest Academic Score %
        student = collection.find_one(sort=[("Academic Score %", -1)], projection={"Student Full Name": 1, "Academic Score %": 1, "_id": 0})
        if student:
            answer = f"{student['Student Full Name']} had the highest score with {student['Academic Score %']}%."
        else:
            answer = "No student data found."
    
    elif "job offer" in question.lower():
        # Fetch the student with the highest job offer value
        student = collection.find_one({"Got Job Offer in Campus Placement": "Yes"}, sort=[("Starting Campus Offer Value", -1)], projection={"Student Full Name": 1, "Starting Campus Offer Value": 1, "_id": 0})
        if student:
            answer = f"{student['Student Full Name']} received the highest job offer with an INR value of {student['Starting Campus Offer Value']}."
        else:
            answer = "No job offer data found."
    
    else:
        answer = f"The LLM model interpreted your query as: {response_text}"

    return jsonify({"question": question, "answer": answer})

# Start the Flask app
if __name__ == '__main__':
    app.run(debug=True)
