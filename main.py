from fastapi import FastAPI
from schemas import ChatRequest

from providers.anthropic import complete



app = FastAPI()

@app.get("/")
def root():
    return {"message": "Azir is running"}


@app.get("/")
def root():
    return {"message": "Azir is running"}


@app.post("/chat")
async def chat(request: ChatRequest):
    return await complete(request)


