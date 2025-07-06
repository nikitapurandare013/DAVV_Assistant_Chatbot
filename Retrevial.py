from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
api_key = os.getenv("GROQ_API_KEY") 


def load_vectorstore():
    """
    Load the FAISS vectorstore using the HuggingFace sentence transformer model.
    
    Returns:
        FAISS: A FAISS vectorstore object loaded with embeddings.
    """
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.load_local(
        folder_path="faiss_store",
        index_name="davv_index",
        embeddings=embedding_model,
        allow_dangerous_deserialization=True
    )
    return vectorstore


def load_llm():
    """
    Load the ChatGroq LLM with the specified Groq API key and model name.
    
    Returns:
        ChatGroq: A Groq language model interface.
    """
    return ChatGroq(
        groq_api_key="groq_api_key",
        model_name="llama-3.3-70b-versatile"
    )


def build_prompt_template():
    """
    Create a custom prompt template to instruct the LLM on how to answer DAVV-related questions.

    Returns:
        PromptTemplate: A LangChain PromptTemplate object.
    """
    return PromptTemplate(
        input_variables=["context", "question"],
        template="""
You are a helpful assistant specialized in providing short, accurate answers about DAVV (Devi Ahilya Vishwavidyalaya) based on the given context. 

Context:
{context}

Question: {question}

Instructions:
- Answer concisely (1-3 sentences).
- Do not add unrelated or extra information.
- If the context does not contain the answer, say "I couldn’t find the answer in the provided information."

Answer:
"""
    )


def setup_qa_chain():
    """
    Set up the RetrievalQA chain using the Groq LLM, FAISS retriever, and custom prompt.

    Returns:
        RetrievalQA: A configured RetrievalQA chain for answering questions.
    """
    retriever = load_vectorstore().as_retriever(search_type="similarity", search_kwargs={"k": 3})
    llm = load_llm()
    prompt = build_prompt_template()
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=False,
        chain_type_kwargs={"prompt": prompt}
    )


def interactive_qa(query):
    qa_chain = setup_qa_chain()
    response = qa_chain.invoke(query)
    return response['result']  

#if __name__ == "__main__":
#   interactive_qa()
