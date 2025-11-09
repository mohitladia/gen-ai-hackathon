from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from tools.retrieval_tool import retrieve_context

def get_retrieval_agent():
    """Create and return the Retrieval Explorer Agent."""
    model = init_chat_model("google_genai:gemini-2.5-flash-lite")
    system_prompt = (
        "You are a Retrieval Explorer Agent. "
        "You can use the 'retrieve_context' tool to fetch and summarize content "
        "from the vector database based on user queries."
    )
    return create_agent(model=model, tools=[retrieve_context], system_prompt=system_prompt)
