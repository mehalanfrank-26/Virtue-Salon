import chromadb
from sentence_transformers import SentenceTransformer
import json

# Load the JSON data
with open("llm_prompts_input.json", "r") as f:
    data = json.load(f)

# Initialize ChromaDB client
client = chromadb.Client()
collection = client.get_or_create_collection("salon_customers")

# Initialize Sentence Transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

for item in data:
    doc_id = str(item["customer_id"])
    # Prepare metadata
    metadata = {
        "name": item["name"],
        "email": item["email"],
        "phone": item["phone"]
    }
    # Prepare document (all other details as a string)
    doc_fields = {k: v for k, v in item.items() if k not in ["customer_id", "name", "email", "phone"]}
    document = json.dumps(doc_fields, ensure_ascii=False)
    # Generate embedding
    embedding = model.encode(document).tolist()
    # Add to ChromaDB
    collection.add(
        ids=[doc_id],
        documents=[document],
        metadatas=[metadata],
        embeddings=[embedding]
    )

print("Data with embeddings stored in ChromaDB!")

# # User-defined query
# query_text = "Customers who did facial 2 months ago"

# # Convert query to embedding
# query_embedding = model.encode(query_text).tolist()

# # Search similar customers
# results = collection.query(
#     query_embeddings=[query_embedding],
#     n_results=5  # Adjust as needed
# )

# # Inspect top matches
# for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
#     print(f"Customer: {meta['name']} - Matched document: {doc}")

import chromadb
from google.generativeai import GenerativeModel
import google.generativeai as genai

# Configure Gemini
genai.configure(api_key="AIzaSyBmsNtAe57aWLVftAEvAvMMEssXbTAzFjo")  # Replace with your Gemini API key
model = genai.GenerativeModel("gemini-1.5-flash")

# Connect to ChromaDB
client = chromadb.Client()
collection = client.get_or_create_collection("salon_customers")

# Ask user for customer name
customer_name_input = input("Enter customer name: ").strip()

# Search ChromaDB for that customer by metadata filtering
results = collection.query(
    query_texts=[" "],  # dummy query; we filter by metadata
    where={"name": customer_name_input},
    n_results=1
)

# Check if we found the customer
if results["documents"][0]:
    document = results["documents"][0][0]  # the appointment data
    metadata = results["metadatas"][0][0]
    customer_name = metadata.get("name", "Customer")

    # Create a personalized prompt
    prompt = f"""
    Generate a friendly salon reminder message for the customer named {customer_name}.
    Here is the appointment info: {document}
    """

    # Use Gemini to generate message
    response = model.generate_content(prompt)
    print("\n📩 Reminder message:")
    print(response.text)
else:
    print(f"❌ No data found for customer '{customer_name_input}'")    
