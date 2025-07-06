from langchain.text_splitter import RecursiveCharacterTextSplitter
import json

# Load your scraped JSON
with open("davv_scraped_data.json", "r", encoding="utf-8") as f:
    raw_data = json.load(f)

# Initialize the chunker
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    separators=["\n\n", "\n", ".", " "]
)

# Create chunked data
chunked_data = []

for doc in raw_data:
    chunks = text_splitter.split_text(doc["content"])
    for chunk in chunks:
        chunked_data.append({
            "url": doc["url"],
            "chunk": chunk
        })

# Save chunked output
with open("davv_chunked_data.json", "w", encoding="utf-8") as f:
    json.dump(chunked_data, f, indent=2, ensure_ascii=False)
