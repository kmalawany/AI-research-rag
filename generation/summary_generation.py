from langchain_core.messages import AIMessage


def choose_paper_prompt(question, context):

    prompt = f"""
                        you will be given a list of papers, each paper contains metadata like id, row, etc
                        you need to choose the most similar paper and then return the id only
                        to the question that will be given to you, you can only choose one document:
                        question: {question}
                        papers: {context}


                        Your answer should only contain the id with quotes nothing else




    """
    return prompt


def summary_prompt(question, context):
    prompt = f"""
            You are an expert in AI research, you will be given a chunks of text from a paper
            and a question. Use the chunks to figure out the answer to the question. you should
            give sufficient and detailed answer using the chunks, you can discard parts of the
            chunks that you believe is irrelevant and only use the relevant parts. 
            
            question: {question}
            chunks: {context}
            
            Instructions: only output the full answer to the question.
            Cite the papers ones in the end after you finish answering.
            Use MLA format, the format is written as follows:
            Last Name, First Name. Title of Book. Publisher, Year
            examples: Beckett, Samuel. The Collected Shorter Plays. Grove Press, 2010.
 

    """
    return prompt


def choose_paper(state, llm):
    docs_content = "\n\n".join(str(doc) for doc in state["context"])
    messages = choose_paper_prompt(state["question"], docs_content)
    response = llm.invoke(messages)
    return {"paper_id": response.content.strip("'").strip('"')}


def generate_answer(state, llm):
    docs_content = "\n\n\n".join(str(doc) for doc in state["context"])
    messages = summary_prompt(state["question"], docs_content)
    response = llm.invoke(messages)
    return {"response": AIMessage(content=response.content, role='end_summary')}
