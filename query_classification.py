from typing import TypedDict, Literal
from model import llm
Category = Literal["topic_search", "summarization/explanation", "out_of_scope"]


class ClassifyQuery(TypedDict):
    category: Category


llm = llm


def classify_prompt(query) -> str:
    prompt = f'''
                You are a classifier for AI research paper queries, classify the user query into one of 
                the following categories:
                - topic_search : includes exploring research areas and general AI related Q&A.
                - summarization/explanation: Includes specific questions about certain topic or Understanding a single paper or group of papers.
                - out_of_scope: queries that are not AI research paper related, any topic that is not AI related is considered out of scope
                                
                
                Examples:
                Show me recent work on diffusion models in computer vision. Expected answer: topic search
                What’s new in reinforcement learning with language models? Expected answer: topic search
                What is contrastive learning? Expected answer: topic search
                tell me about large vision models Expected answer: topic search
                Tell me examples of AI being used by economists Expected answer: topic search
                Explain the attention mechanism with examples. Expected answer: Summarization / Explanation
                tell me about the architecture of SAM(segment anything) model Expected answer: Summarization / Explanation
                Summarize Distributionally Robust Receive Combining paper in simple terms Expected answer: Summarization / Explanation
                Can you explain the loss function described in the appendix? Expected answer: Summarization / Explanation
                What’s the weather in Paris today? Expected answer: out of scope
                Who won the football match last night? Expected answer: out of scope
                Write me a poem about AI in Shakespeare style. Expected answer: out of scope
                Can you help me debug my Python code? Expected answer: out of scope
                in Denoising Diffusion Probabilistic Models paper: What are diffusion probabilistic models, and how do they generate images? Expected answer: Summarization / Explanation
                show me examples of skin cancer detection models Expected answer: topic search

                
                Query: {query}
                
                instructions: respond with EXACTLY one of these three tokens (no punctuation, no extra words):
                - topic_search
                - summarization_explanation
                - out_of_scope
    '''

    return prompt


def classify_query(state) -> ClassifyQuery:
    query = state['question']
    prompt = classify_prompt(query)
    response = llm.invoke(prompt)
    category = response.content.strip().lower()
    return {'category': category}


