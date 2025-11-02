
from dotenv import load_dotenv
from fastapi import FastAPI
from agent import query_rag_agent
from fastapi.middleware.cors import CORSMiddleware
from models.chat_models import ChatModel
load_dotenv()

app = FastAPI(
    title="RAG Agent API",
    description="API to interact with the RAG agent for querying information about Aman Obaid.",
    version="1.0.0"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/query", summary="Query the RAG Agent", description="Send a question to the RAG agent and receive a response.")
def query_agent_endpoint(request: ChatModel):
    """API endpoint to query the RAG agent."""
    response = query_rag_agent(request.query, thread_id=request.thread_id)
    return {"response": response["messages"][-1].content}
   
import uvicorn  
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
            