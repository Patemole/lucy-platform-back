from pydantic import BaseModel

class InputQuery(BaseModel):
    # course_name: str # TO BE DEPRECATED
    course_id: str
    chat_id: str
    username: str
    message: str
        