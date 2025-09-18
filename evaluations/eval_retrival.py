import argparse
from dotenv import load_dotenv
from tqdm import tqdm
import logging
from model import llm
from ast import literal_eval
from evaluations.data_preparation_retriever import load_prepare_dataset
from evaluations.metrics import recall_at_k, precision_at_k

load_dotenv()


def eval_prompt(question, context, ground_truth):

    prompt = f"""
        You are an evaluator for research paper retrieval.

        You will be given:
        - question: a user’s research question
        - ground_truth: one or more documents known to be relevant
        - context: a list of 10 retrieved documents
        
        Task:
        For each retrieved document in the list, decide if it is relevant to the question.
        - A document is RELEVANT if it helps answer the question in a way similar to the ground_truth documents.
        - A document is IRRELEVANT if it does not help answer the question.
        - return 1 if RELEVANT and 0 if IRRELEVANT you should do that for each document nothing else
        
        Output:
        Return ONLY a list of 10 digits (0 or 1), in the same order as the context documents.
        Example: [1,0,1,0,0,1,0,0,0,1]
        ONLY return a list of 10 digits (0 or 1) nothing else. dont explain you answer. dont give context
        
        question: {question}
        context: {context}
        ground_truth: {ground_truth}
"""
    return prompt


def parse_output(output):
    try:
        parsed = literal_eval(output.strip())
        if isinstance(parsed, list) and all(isinstance(x, int) for x in parsed):
            return parsed
        else:
            raise ValueError('Output is not a list of ints')

    except Exception as e:
        logging.error("Parsing error:", e)
        return []


def clean_output(test_output):
    eval_results = []
    #output_lst = literal_eval(test_output)
    for i in test_output:
        if i:
            eval_results.append(i)
    return eval_results


def evaluate_retriever(llm, test_set):
    test_output = []

    for d in tqdm(test_set, desc='Evaluating retriever'):
        prompt = eval_prompt(d['query'], d['context'], d['ground_truth'])
        response = llm.invoke(prompt)
        output = parse_output(response.content)
        test_output.append(output)
    cleaned_output = clean_output(test_output)
    return cleaned_output


def main():

    ap = argparse.ArgumentParser()
    ap.add_argument('query_path', help='queries folder path')
    ap.add_argument('corpus_path', help='corpus_clean path')
    ap.add_argument('k', help='k value')

    args = ap.parse_args()

    query_path = args.query_path
    corpus_path = args.corpus_path
    k = args.k
    dataset = load_prepare_dataset(query_path,
                                   corpus_path)
    test_output = evaluate_retriever(llm=llm, test_set=dataset)
    precision = precision_at_k(test_output, k)
    recall = recall_at_k(test_output, k)
    print("Precision@", k, ': ', precision)
    print("Recall@", k, ': ', recall)


if __name__ == '__main__':
    main()