from pymongo import MongoClient
from sentence_transformers import SentenceTransformer, util
import numpy as np
import torch
import json
from bson import ObjectId
from datetime import datetime

# Connect to MongoDB
client = MongoClient('mongodb://localhost:27017/')
db = client['alumni_db']
collection = db['alumni_data']

# Load the pre-trained model for embeddings
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to generate embedding for a query
def generate_query_embedding(query):
    query_embedding = embedding_model.encode(query, convert_to_tensor=True)
    return query_embedding

# Function to retrieve alumni data based on a user query
def retrieve_alumni_data(query):
    query_embedding = generate_query_embedding(query)
    
    # Retrieve all alumni records with embeddings
    alumni_records = list(collection.find({"embedding": {"$exists": True}}))
    
    if not alumni_records:
        return "No alumni records found."
    
    # Convert stored embeddings to tensors
    embeddings = [np.array(record['embedding']) for record in alumni_records]
    embeddings = torch.tensor(embeddings, dtype=torch.float32)
    
    # Compute cosine similarities
    similarities = util.pytorch_cos_sim(query_embedding, embeddings)
    
    # Find the most similar record
    most_similar_idx = torch.argmax(similarities).item()
    most_similar_record = alumni_records[most_similar_idx]
    
    # Format the result as a sentence
    formatted_result = (
        f"Student Full Name: {most_similar_record.get('Student Full Name', 'N/A')}. "
        f"Department: {most_similar_record.get('Department Name/Code', 'N/A')}. "
        f"Joined in {most_similar_record.get('Joining Year', 'N/A')} and graduated in {most_similar_record.get('Graduation Year', 'N/A')}. "
        f"Email: {most_similar_record.get('Contact Email', 'N/A')}. "
        f"Phone Number: {most_similar_record.get('Contact Phone Number', 'N/A')}. "
        f"Date of Birth: {most_similar_record.get('Date of Birth', 'N/A')}. "
        f"Address: {most_similar_record.get('Student Address', 'N/A')}, {most_similar_record.get('Student City', 'N/A')}. "
        f"Academic Score: {most_similar_record.get('Academic Score %', 'N/A')}%. "
        f"Attendance: {most_similar_record.get('Attendance %', 'N/A')}%. "
        f"Job Offer in Campus Placement: {most_similar_record.get('Got Job Offer in Campus Placement', 'N/A')}. "
        f"Company: {most_similar_record.get('Job Offered by Company', 'N/A')}. "
        f"Offer Value: ${most_similar_record.get('Starting Campus Offer Value', 'N/A')}. "
        f"Notes: {most_similar_record.get('Notes', 'N/A')}."
    )
    
    return formatted_result

# Function to serialize MongoDB objects (ObjectId, datetime) to JSON-compatible format
def mongo_to_json(data):
    if isinstance(data, dict):
        return {key: mongo_to_json(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [mongo_to_json(item) for item in data]
    elif isinstance(data, ObjectId):
        return str(data)
    elif isinstance(data, datetime):
        return data.isoformat()
    else:
        return data

# Main function to get input from the terminal and print output
if __name__ == '__main__':
    user_query = input("Enter your query: ")
    
    # Retrieve relevant alumni data
    result = retrieve_alumni_data(user_query)
    
    # Convert MongoDB ObjectId and datetime fields to JSON-serializable format
    result = mongo_to_json(result)

    # Print the result
    print(result)
