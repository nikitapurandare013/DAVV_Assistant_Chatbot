from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

# Load tokenizer and model from HuggingFace
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

# Function to generate sentence embedding
def get_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    # Use the [CLS] token's output as the sentence embedding
    cls_embedding = outputs.last_hidden_state[:, 0, :]
    return cls_embedding

# Function to compute cosine similarity between two sentences
def get_similarity(sent1, sent2):
    emb1 = get_embedding(sent1)
    emb2 = get_embedding(sent2)
    similarity = F.cosine_similarity(emb1, emb2)
    return similarity.item()

# Test Example
sentence1 = "I love machine learning"
sentence2 = "I enjoy studying artificial intelligence"

similarity_score = get_similarity(sentence1, sentence2)
print(f"Cosine Similarity: {similarity_score:.4f}")
