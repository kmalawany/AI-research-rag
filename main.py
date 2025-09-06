from fastapi import FastAPI
from graph import graph
from pydantic import BaseModel


app = FastAPI()


class ChatInput(BaseModel):
    question: str
    thread_id: str


@app.get("/")
async def root():
    return {"msg": "FastAPI is alive"}


@app.post(path="/chat")
async def chat(user_input: ChatInput):
    config = {"configurable": {"thread_id": user_input.thread_id}}
    response = await graph.ainvoke({"question": user_input.question}, config=config)
    return response['response'].content

