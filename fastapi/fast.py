
from fastapi import FastAPI
from pydantic import BaseModel
app = FastAPI()
class UserResponse(BaseModel):
    user_text: str
app = FastAPI()

@app.post('/getChat')
async def get_ChatAnswer(user_data:UserResponse):
    response = user_data.user_text
    return response

    