import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.docstore.document import Document


def chunking(input_file="davv_scraped_data.json",
             output_file="davv_chunked_data.json",
             chunk_size=500,
             chunk_overlap=100,
             verbose=True):
    """
    Loads a scraped JSON file, chunks the content using RecursiveCharacterTextSplitter,
    and saves the chunked output to a new JSON file.
    """

    try:
        with open(input_file, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to load input file: {e}")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " "]
    )

    chunked_data = [
        {"url": doc["url"], "chunk": chunk}
        for doc in raw_data
        for chunk in text_splitter.split_text(doc.get("content", ""))
    ]

    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(chunked_data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        raise RuntimeError(f"Failed to save chunked output: {e}")

    if verbose:
        print(f"✅ Chunking complete. Output saved to '{output_file}'")

    return chunked_data

def create_faiss_index_langchain(chunk_file="davv_chunked_data.json",
                                  save_dir="faiss_store",
                                  index_name="davv_index",
                                  model_name="sentence-transformers/all-MiniLM-L6-v2",
                                  verbose=True):
    """
    Creates and saves a FAISS vector index using LangChain from a JSON file of text chunks.

    Args:
        chunk_file (str): Path to the JSON file containing preprocessed text chunks.
                          Each item should have a "chunk" field.
        save_dir (str): Directory path to save the FAISS index files (.faiss and .pkl).
        index_name (str): Name to assign to the FAISS index.
        model_name (str): Name of the Hugging Face sentence transformer model to use for embeddings.
        verbose (bool): If True, prints status messages during the process.

    Returns:
        FAISS: A LangChain FAISS vectorstore object containing the embedded documents.

    Raises:
        RuntimeError: If the input chunk file cannot be read or parsed.
        ValueError: If no valid text chunks are found for embedding.
    """
    try:
        with open(chunk_file, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to load chunk file: {e}")

    documents = [entry["chunk"].strip() for entry in raw_data if entry.get("chunk") and entry["chunk"].strip()]

    if verbose:
        print(f"✅ Total valid chunks found: {len(documents)}")

    if not documents:
        raise ValueError("❌ No valid chunks found to embed!")

    if verbose:
        print("🚀 Loading embedding model...")
    embedding_model = HuggingFaceEmbeddings(model_name=model_name)

    if verbose:
        print("📍 Creating LangChain Document objects...")
    langchain_docs = [Document(page_content=doc) for doc in documents]

    if verbose:
        print("🔧 Creating FAISS index using LangChain...")
    vectorstore = FAISS.from_documents(langchain_docs, embedding_model)

    if verbose:
        print(f"💾 Saving index to '{save_dir}' with name '{index_name}'...")
    vectorstore.save_local(save_dir, index_name=index_name)

    if verbose:
        print("✅ FAISS index (.faiss + .pkl) saved successfully.")

    return vectorstore 
      
