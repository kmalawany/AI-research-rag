from langchain import hub
from langchain_core.messages import AIMessage

# prompt = hub.pull("rlm/rag-prompt")
# example_message = prompt.invoke(
#     {"context": "(context goes here)", "question": "(question goes here)"}
# ).to_messages()


def topic_prompt(question, context):
    prompt = f""""
            You are a search engine for AI research papers. You will be given a question about
            a certain topic and some papers related to the question, use the papers to get information
            about the question, and give back a clear answer.
            question: {question}
            context: {context}
            
            Instructions: give back the answer to the question and cite the answer with the papers you
            used to answer the questions, but don't tell that you are using papers or chunks of text.
             Use MLA format, the format is written as follows:
            Last Name, First Name. Title of Book. Publisher, Year
            examples: Beckett, Samuel. The Collected Shorter Plays. Grove Press, 2010.
            
            Cite the papers ones in the end after you finish answering
            
    """""

    return prompt


def generate_topic(state, llm):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    #messages = prompt.invoke({"question": state["question"], "context": docs_content})
    messages = topic_prompt(state["question"], docs_content)
    response = llm.invoke(messages)
    #response.content for ollama
    return {"response": AIMessage(content=response.content, role='topics')}


