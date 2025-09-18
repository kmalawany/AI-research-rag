import pandas as pd
import json
from retrieval.topic_retrieval import retrieve_topic
from tqdm import tqdm
from dotenv import load_dotenv


load_dotenv()


def load_prepare_dataset(query_dir: str, corpus_dir: str):
    df_query = pd.read_parquet(query_dir)
    df_corpus = pd.read_parquet(corpus_dir)

    df_query = df_query.explode('corpusids')
    df_query = df_query.rename(columns={'corpusids': 'corpusid'})
    df_query['corpusid'] = df_query['corpusid'].astype(int)

    df_merged = df_query.merge(df_corpus[["corpusid", "title", "abstract"]],
                               on='corpusid')
    df_merged = df_merged[["corpusid", "query", "title", "abstract"]]
    df_merged['title_abstract'] = df_merged['title'] + " \n \n " + df_merged['abstract']
    df_merged.drop(columns=['title', 'abstract'], inplace=True)
    dataset = df_merged.groupby("query")["title_abstract"].apply(list).to_dict()

    testcases = []

    for query, truth in tqdm(dataset.items(), desc='Preparing dataset'):
        docs = retrieve_topic({'question': query})

        testcases.append({
            'query': query,
            'context': [doc.page_content for doc in docs['context']],
            'ground_truth': truth
        })

    with open('testset.json', 'w', encoding="utf-8") as f:
        json.dump(testcases, f, indent=2, ensure_ascii=False)

    return testcases
