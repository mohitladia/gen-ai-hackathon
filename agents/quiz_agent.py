import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from tools.retrieval_tool import retrieve_context


def get_quiz_agent():
    """
    Create and return a Quiz Generation Agent.

    The agent:
    - Uses Gemini (google_genai:gemini-2.5-flash-lite)
    - Can call the `retrieve_context` tool to fetch factual info from your Chroma vector DB
    - Generates conceptual MCQs with marked correct answers
    """
    # Load environment variables (for Google API key)
    load_dotenv()

    # Initialize the Gemini model
    model = init_chat_model("google_genai:gemini-2.5-flash-lite")

    # Register the retrieval tool
    tools = [retrieve_context]

    # Define the system prompt
    system_prompt = (
        "You are a Quiz Generation Assistant. "
        "You have access to a tool named 'retrieve_context' that retrieves factual information "
        "from educational PDFs stored in a vector database. "
        "When a user gives a topic, use that tool to fetch context and generate exactly "
        "5 multiple-choice questions (MCQs) with 4 options each. "
        "Clearly mark the correct answer with ✅. "
        "Ensure all questions are conceptual, accurate, and based on the retrieved content."
    )

    # Create and return the agent
    return create_agent(
        model=model,
        tools=tools,
        system_prompt=system_prompt
    )
