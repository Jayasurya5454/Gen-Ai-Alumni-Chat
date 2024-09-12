from flask import Flask, request, jsonify
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer, util
import numpy as np
import torch

app = Flask(__name__)

# Connect to MongoDB
client = MongoClient('mongodb://localhost:27017/')
db = client['alumni_db']
collection = db['alumni_data']

# Load the pre-trained model for embeddings
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def generate_query_embedding(query):
    # Generate embedding for the query
    query_embedding = embedding_model.encode(query, convert_to_tensor=True)
    return query_embedding

def retrieve_alumni_data(query):
    # Generate embedding for the query
    query_embedding = generate_query_embedding(query)
    
    # Retrieve all alumni records with embeddings
    alumni_records = list(collection.find({"embedding": {"$exists": True}}))
    
    if not alumni_records:
        return {"message": "No alumni records found."}
    
    # Convert stored embeddings to tensors
    embeddings = [np.array(record['embedding']) for record in alumni_records]
    embeddings = torch.tensor(embeddings, dtype=torch.float32)  # Convert to PyTorch tensors
    
    # Compute cosine similarities
    similarities = util.pytorch_cos_sim(query_embedding, embeddings)
    
    # Find the most similar record
    most_similar_idx = torch.argmax(similarities).item()  # Use torch.argmax and convert to item
    most_similar_record = alumni_records[most_similar_idx]
    
    return most_similar_record

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    user_input = data.get('query', '')
    
    if not user_input:
        return jsonify({"error": "Query parameter is missing"}), 400
    
    # Retrieve relevant alumni data
    relevant_alumni_data = retrieve_alumni_data(user_input)
    
    return jsonify(relevant_alumni_data)

if __name__ == '__main__':
    app.run(debug=True)
