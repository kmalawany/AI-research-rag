from typing import Annotated, Optional, Literal, List, TypedDict
from langchain_ollama import ChatOllama
from langchain_google_genai import GoogleGenerativeAI
from pprint import pprint
from langgraph.graph import StateGraph, START, END
from query_classification import classify_query
from langchain_core.documents import Document
from retrieval.topic_retrieval import retrieve_topic
from retrieval.summary_retrieval import summary_retrieve
from generation.topic_generation import generate_topic
from generation.summary_generation import choose_paper, generate_answer
from langchain_core.messages import AIMessage
from pdf.pdf_handling import get_paper, generate_paper_embeddings
from state import State
from dotenv import load_dotenv

load_dotenv()
llm = ChatOllama(model="llama3.2:3b", temperature=0)


def decide_agent(state) -> Literal['summary', 'topic_search', 'out_of_context']:
    if state['category'] == 'summarization/explanation':
        return 'summary'
    elif state['category'] == 'topic_search':
        return 'topic_search'
    else:
        return 'out_of_context'


def clarification_node(state, category):
    print('Do you mean' + category + '?')
    return state


def out_of_context(state) -> dict:
    return {'response': AIMessage(content='This query is out of context', role='end')}


builder = StateGraph(state_schema=State)  # type: ignore[arg-type]
builder.add_node('classify', classify_query)
builder.add_node('out_of_context', out_of_context)
builder.add_node('topic_search', retrieve_topic)
builder.add_node('summary', summary_retrieve)
builder.add_node('choose_paper', lambda state: choose_paper(state, llm=llm))
builder.add_node('generate_topics', lambda state: generate_topic(state, llm=llm))
builder.add_node('download_paper', get_paper)
builder.add_node('generate_papers_embedding', generate_paper_embeddings)
builder.add_node('generate_answer', lambda state: generate_answer(state, llm=llm))

builder.add_edge(START, 'classify')
builder.add_conditional_edges('classify', decide_agent)
builder.add_edge('topic_search', 'generate_topics')
builder.add_edge('summary', 'choose_paper')
builder.add_edge('choose_paper', 'download_paper')
builder.add_edge('download_paper', 'generate_papers_embedding')
builder.add_edge('generate_papers_embedding', 'generate_answer')
builder.add_edge('generate_answer', END)
builder.add_edge('generate_topics', END)

builder.add_edge('out_of_context', END)

graph = builder.compile()

