import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Load chunked data
with open("davv_chunked_data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Extract and clean text chunks
chunks = [item["content"].strip() for item in data if "content" in item and item["content"].strip() != ""]

print(f"✅ Total valid chunks found: {len(chunks)}")

if not chunks:
    raise ValueError("❌ No valid chunks found to embed!")

# Load SentenceTransformer model
print("🚀 Loading model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

# Generate embeddings
print("🧠 Generating embeddings...")
embeddings = model.encode(chunks, convert_to_numpy=True, show_progress_bar=True)

print("✅ Embeddings shape:", embeddings.shape)

if embeddings.shape[0] == 0:
    raise ValueError("❌ No embeddings generated!")

# Convert to float32 for FAISS
embeddings = embeddings.astype("float32")

# Create FAISS index
print("📦 Creating FAISS index...")
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# Save index
faiss.write_index(index, "davv_faiss_index.idx")
print("✅ FAISS index saved as 'davv_faiss_index.idx'")



