from typing import Annotated, Optional, Literal, List, TypedDict
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, AnyMessage
from langgraph.graph import add_messages
from pymupdf import Document as do


class State(TypedDict):
    question: str
    category: Optional[str]
    response: AIMessage
    context: List[Document]
    paper_id: float
    paper_text: List[Document]

    messages_hist: Optional[Annotated[list[AnyMessage], add_messages]]
