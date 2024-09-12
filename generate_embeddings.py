from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
import numpy as np
import json

# Connect to MongoDB
client = MongoClient('mongodb://localhost:27017/')
db = client['alumni_db']
collection = db['alumni_data']

# Load the pre-trained model for embeddings
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def generate_and_store_embeddings():
    # Retrieve all alumni records
    alumni_records = list(collection.find({}))
    
    for record in alumni_records:
        # Create a description string from the record
        description = (
            f"Name: {record['Student Full Name']}, Department: {record['Department Name/Code']}, "
            f"Graduation Year: {record['Graduation Year']}, City: {record['Student City']}, "
            f"Job Offer: {record['Got Job Offer in Campus Placement']}, Notes: {record['Notes']}"
        )
        
        # Generate embedding for the description
        embedding = embedding_model.encode(description, convert_to_tensor=True)
        
        # Convert embedding to a list for storage
        embedding_list = embedding.tolist()
        
        # Update the record with embedding
        collection.update_one(
            {"_id": record["_id"]},
            {"$set": {"embedding": embedding_list}}
        )
    
    print("Embeddings generated and stored successfully.")

if __name__ == '__main__':
    generate_and_store_embeddings()
