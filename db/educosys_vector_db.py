from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings


def get_vector_store():
    """
    Return a handle to the existing Chroma vector store.

    This function connects to the Chroma DB that was previously built and
    persisted using `build_vector_db.py`. It does NOT reload or embed any
    new documents — it simply provides a reference for reading and querying.

    Returns:
        Chroma: A Chroma vector store instance connected to the persisted DB.
    """
    # Initialize the same embedding model used during DB creation
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

    # Connect to the existing persisted vector store
    vector_store = Chroma(
        collection_name="educosys_learning_material",
        embedding_function=embeddings,
        persist_directory="./chroma_educosys_db"  # Path where DB was saved
    )

    return vector_store
