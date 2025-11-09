import logging
from langchain.tools import tool
from db.educosys_vector_db import get_vector_store

logger = logging.getLogger(__name__)

@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """
    Retrieve relevant information from the persisted Chroma vector store.

    Args:
        query (str): User query or topic to search for.

    Returns:
        tuple:
            - str: A serialized text summary combining metadata and content 
                   (for LLM context)
            - list[Document]: The actual retrieved Document objects from the store
    """
    try:
        vector_store = get_vector_store()
        retrieved_docs = vector_store.similarity_search(query, k=2)

        if not retrieved_docs:
            logger.warning(f"No documents found for query: '{query}'")
            return "No relevant content found in the knowledge base.", []

        serialized = "\n\n".join(
            f"Source: {doc.metadata.get('source', 'unknown')}\nContent: {doc.page_content}"
            for doc in retrieved_docs
        )

        logger.info(f"Retrieved {len(retrieved_docs)} docs for query: '{query}'")
        return serialized, retrieved_docs

    except Exception as e:
        logger.error(f"Error retrieving context: {e}")
        return f"Error retrieving context: {str(e)}", []
