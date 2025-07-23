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