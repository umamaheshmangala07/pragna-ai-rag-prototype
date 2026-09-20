"""
pragna-ai-rag-prototype
------------------------
Retrieval-Augmented Generation (RAG) pipeline for PragnaAI.

Fixes vs. the original prototype:
  1. Multilingual embeddings (Telugu-capable) instead of English-only MiniLM.
  2. An actual generation step (Azure OpenAI) that turns retrieved chunks
     into a real answer, not just raw text fragments.
  3. Works on either a PDF file OR a list of plain-text passages, so you
     can test it today without needing a PDF ready.
  4. Persisted vector store so you don't re-embed everything every run.
  5. A test_retrieval() that actually retrieves something, using a real
     Telugu farming question against sample agricultural advisory text.

Setup:
    pip install -r requirements.txt

    Optional (only needed if you want real generated answers instead of
    just raw retrieved chunks): set these environment variables with your
    Azure OpenAI project details from https://ai.azure.com
        AZURE_OPENAI_ENDPOINT      e.g. https://<resource>.openai.azure.com/
        AZURE_OPENAI_API_KEY
        AZURE_OPENAI_DEPLOYMENT    e.g. gpt-4o-mini

    Without those set, the pipeline still runs and returns the raw
    retrieved passages, clearly labeled, so you can test retrieval quality
    on its own first.
"""

import os
from typing import List, Optional

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document

# Multilingual embedding model: understands Telugu (and 50+ other
# languages) reasonably well, unlike the English-only MiniLM model used
# in the original version of this script. Still free and open-source.
MULTILINGUAL_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

PERSIST_DIR = "./pragna_vectorstore"


def _load_documents_from_pdf(pdf_path: str) -> List[Document]:
    """Load and split a PDF into retrievable chunks."""
    loader = PyPDFLoader(pdf_path)
    documents = loader.load_and_split()
    print(f"Loaded and split {len(documents)} chunks from {pdf_path}.")
    return documents


def _load_documents_from_texts(texts: List[str]) -> List[Document]:
    """Wrap plain-text passages as Documents (no PDF needed)."""
    documents = [Document(page_content=t) for t in texts]
    print(f"Loaded {len(documents)} plain-text passages.")
    return documents


def initialize_rag_pipeline(
    pdf_path: Optional[str] = None,
    texts: Optional[List[str]] = None,
    persist_directory: str = PERSIST_DIR,
):
    """
    Build a retriever from either a PDF file or a list of text passages.

    Exactly one of pdf_path or texts should be provided.
    """
    if pdf_path:
        print(f"Initializing RAG pipeline for document: {pdf_path}")
        try:
            documents = _load_documents_from_pdf(pdf_path)
        except Exception as e:
            print(f"Error loading document: {e}")
            return None
    elif texts:
        print("Initializing RAG pipeline for in-memory text passages.")
        documents = _load_documents_from_texts(texts)
    else:
        print("Error: provide either pdf_path or texts.")
        return None

    # Multilingual embeddings so Telugu content is actually understood,
    # not just English.
    embeddings = HuggingFaceEmbeddings(model_name=MULTILINGUAL_EMBEDDING_MODEL)

    # Persisted vector store: reuse across runs instead of re-embedding
    # everything every single time.
    vectorstore = Chroma.from_documents(
        documents,
        embeddings,
        persist_directory=persist_directory,
    )
    vectorstore.persist()

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    return retriever


def generate_answer(question: str, retrieved_docs: List[Document]) -> str:
    """
    Turn retrieved chunks + the question into an actual answer using
    Azure OpenAI, if credentials are configured. Falls back to returning
    the raw retrieved context if they aren't, so the pipeline is still
    runnable and testable without any API keys.
    """
    context = "\n\n".join(d.page_content for d in retrieved_docs)

    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
    api_key = os.environ.get("AZURE_OPENAI_API_KEY")
    deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT")

    if not (endpoint and api_key and deployment):
        return (
            "[No Azure OpenAI credentials configured — showing raw "
            "retrieved context instead of a generated answer]\n\n"
            f"{context}"
        )

    try:
        from openai import AzureOpenAI

        client = AzureOpenAI(
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version="2024-08-01-preview",
        )

        system_prompt = (
            "You are PragnaAI's assistant. Answer the user's question "
            "using ONLY the context provided below. If the context "
            "doesn't contain the answer, say so honestly rather than "
            "guessing. Respond in the same language as the question."
        )

        response = client.chat.completions.create(
            model=deployment,
            messages=[
                {"role": "system", "content": f"{system_prompt}\n\nContext:\n{context}"},
                {"role": "user", "content": question},
            ],
            temperature=0.3,
            max_tokens=400,
        )
        return response.choices[0].message.content

    except Exception as e:
        return f"[Generation failed: {e}]\n\nRaw retrieved context:\n{context}"


def ask(retriever, question: str) -> str:
    """Retrieve relevant chunks for a question, then generate an answer."""
    docs = retriever.invoke(question)
    return generate_answer(question, docs)


def test_retrieval():
    """
    A real, working test: builds a tiny sample knowledge base of
    agricultural advisory content (English + Telugu), then asks a
    farming question in Telugu and shows what comes back.
    """
    sample_agri_passages = [
        # English sample passage
        "Groundnut crops in Andhra Pradesh should be sown at the onset "
        "of monsoon, typically mid-June to early July, for best yield. "
        "Ensure soil moisture is adequate before sowing.",
        # Telugu sample passage (groundnut sowing guidance)
        "వేరుశనగ పంటను ఆంధ్రప్రదేశ్‌లో వర్షాకాలం ప్రారంభంలో, జూన్ మధ్య నుండి "
        "జూలై మొదటి వారంలో విత్తడం మంచిది. మంచి దిగుబడి కోసం విత్తే ముందు "
        "నేలలో తగినంత తేమ ఉండేలా చూసుకోవాలి.",
        # A second, unrelated Telugu passage to confirm retrieval actually
        # picks the *relevant* one, not just any Telugu text.
        "వరి పంటకు నీరు ఎక్కువగా అవసరం. నాటిన తర్వాత పొలంలో నిరంతరం నీరు "
        "నిలిచి ఉండేలా చూసుకోవాలి.",
    ]

    print("Building sample knowledge base (English + Telugu agri content)...")
    retriever = initialize_rag_pipeline(texts=sample_agri_passages)

    if retriever is None:
        print("Failed to initialize retriever.")
        return

    telugu_question = "వేరుశనగ ఎప్పుడు విత్తాలి?"  # "When should groundnut be sown?"
    print(f"\nTest question (Telugu): {telugu_question}")
    print("(translation: 'When should groundnut be sown?')\n")

    answer = ask(retriever, telugu_question)
    print("Answer:")
    print(answer)


if __name__ == "__main__":
    test_retrieval()

    # Example usage with a real PDF once you have one, e.g. state
    # agribusiness support and incubation guidelines:
    # retriever = initialize_rag_pipeline(pdf_path="AP_startup_subsidy_guidelines.pdf")
    # print(ask(retriever, "What subsidies are available for drip irrigation?"))
