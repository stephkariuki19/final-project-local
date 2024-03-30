
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()
class UserResponse(BaseModel):
    user_text: str

@app.get('/')
async def root():
    return {'example':'jacob'}

    

@app.post('/getChat')
async def getInput( user_response: UserResponse):
     query  = user_response.user_text
     return query
    