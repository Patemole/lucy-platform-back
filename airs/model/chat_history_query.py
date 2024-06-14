from pydantic import BaseModel

class ChatHistoryQuery(BaseModel):
    chat_id: str