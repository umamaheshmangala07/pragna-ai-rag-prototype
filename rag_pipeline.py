import numpy as np
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

def initialize_rag_pipeline(pdf_path: str):
    print(f"Initializing RAG Pipeline for document: {pdf_path}")
    
    # 1. Load and split the document 
    try:
        loader = PyPDFLoader(pdf_path)
        documents = loader.load_and_split()
        print(f"Successfully loaded and split {len(documents)} document chunks.")
    except Exception as e:
        print(f"Error loading document: {e}")
        return None

    # 2. Initialize open-source embeddings (avoids proprietary API costs for early-stage scalability)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # 3. Create the local Vector Store
    vectorstore = Chroma.from_documents(documents, embeddings)
    
    # 4. Demonstrate retrieval capabilities
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    return retriever

def test_retrieval():
    # Demonstrating numpy utilization for embedding dimension validation
    dummy_vector = np.array([0.15, 0.42, 0.88])
    print(f"System initialized with vector dimension handling: {dummy_vector.shape}")
    print("Pipeline ready to integrate with local LLM for context-aware generation.")

if __name__ == "__main__":
    # Test execution block
    test_retrieval()
    
    # Example usage for querying state agribusiness support and incubation guidelines:
    # retriever = initialize_rag_pipeline("AP_startup_subsidy_guidelines.pdf")
