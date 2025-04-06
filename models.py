from typing import Optional
from pydantic import BaseModel, Field

class AskRequest(BaseModel):
    """
    The shape of JSON that the client must send to /ask.
    """
    prompt: str = Field(..., description="User's query or message.")
    session_id: Optional[str] = Field(None, description="Unique session ID if you want to track conversation across requests.")

class AskResponse(BaseModel):
    """
    The shape of JSON we return. 
    """
    session_id: str
    prompt: str
    response: str
    response_time: float