from pydantic import BaseModel, Field

class ChatModel(BaseModel):
    """Base class for chat models."""
    query: str = Field(..., description="The input query for the chat model.")
    thread_id: str = Field("1", description="The thread ID for the chat session.")

