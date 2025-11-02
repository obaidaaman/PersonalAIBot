

from langchain.tools import tool
from langchain_community.tools import TavilySearchResults
from langchain_tavily import TavilySearch
from langchain.agents.middleware import SummarizationMiddleware
from langchain_openai import ChatOpenAI
import chromadb
import os
from langchain.agents import create_agent
from dotenv import load_dotenv
from langgraph.checkpoint.memory import InMemorySaver
load_dotenv()

checkpoint_saver = InMemorySaver()
chroma_client = chromadb.CloudClient(
    api_key=os.getenv("CHROMA_API_KEY"),  
    tenant=os.getenv("CHROMA_TENANT"),
    database=os.getenv("CHROMA_DATABASE")
)

search_tool = TavilySearch()
@tool
def retrieve_context(query: str, n_results: int = 3) -> str:
    """
    Retrieve relevant documents from ChromaDB based on the query.
    
    Args:
        query: The search query string
        n_results: Number of top results to retrieve (default: 5)"""
    
    try:
        collection = chroma_client.get_collection(name="context_data")

        results = collection.query(
            query_texts=[query],
            n_results=n_results
        )
        if results['documents'] and results['documents'][0]:
            docs = results['documents'][0]
            
            return "\n\n".join(docs)
        else:
            return "No documents found for the given query."
    except Exception as e:
        return f"Error accessing ChromaDB collection: {e}"
    


tools = [retrieve_context, search_tool]

agent = create_agent(
    model = "openai:gpt-4o",
    tools = tools,
    middleware= [SummarizationMiddleware(
        model= "openai:gpt-4o",
        max_tokens_before_summary=2000,
        messages_to_keep=11
    )],
    checkpointer=InMemorySaver(),
    system_prompt = """

You are a highly professional personal assistant representing Aman Obaid in all conversations.

Your role is to communicate with  HR professionals, recruiters, and hiring managers on Aman's behalf.

Maintain a polished, confident, and executive tone. Always present Aman as a capable, well-qualified, and well-rounded professional.

use RAG tool to fetch context about Aman before answering questions.
CRITICAL: For contact information, ONLY use the 'retrieve_context' tool and NEVER use the search tool. The verified contact information is available in the retrieved context.
You should:
- Speak in the first person as Aman's representative (e.g., “Aman has experience in…” or “He has led projects involving…”).
- Use the 'retrieve_context' tool to gather verified and relevant background data before answering.
- Provide clear, structured, and professional answers to any questions about Aman’s background, skills, work experience, or accomplishments.
- If specific details are missing, respond gracefully by emphasizing Aman’s adaptability and strengths.
- Never use filler phrases like “I’m an AI assistant.” Instead, act naturally as his real personal assistant.


Share contact info of Aman Obaid when asked.
Your tone should reflect professionalism, warmth, and confidence — like an executive assistant managing professional correspondence for a high-performing individual. provide crisp and only the crux of the information.
"""
)



def query_rag_agent(question: str, thread_id : str) -> str:
    """Function to query the React agent with a question."""
    try:

        response = agent.invoke({"messages": [{"role": "user", "content": question}]},
                                {"configurable": {"thread_id": thread_id}},)
        return response
    except Exception as e:
        return f"Error querying agent: {e}"


