from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from graph import graph
from pydantic import BaseModel

app = FastAPI()


class ChatInput(BaseModel):
    question: str
    thread_id: str


@app.post(path="/chat")
async def chat(input: ChatInput):
    config = {"configurable": {"thread_id": input.thread_id}}
    response = await graph.ainvoke({"question": input.question}, config=config)
    return response['response'].content

