import argparse
import json
import pandas as pd
import re
from tqdm import tqdm
from embeddings.data_embedding import load_data, pgvector
from dotenv import load_dotenv
import os

load_dotenv()


def convert_to_df(file_path):
    records = []
    year = 0
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            paper = json.loads(line)
            journal_ref = str(paper.get("journal-ref", None))

            if journal_ref:
                match = re.search(r"\b(19|20)\d{2}\b", journal_ref)
                if match:
                    year = int(match.group(0))
            else:
                match_new = re.match(r'arXiv:(\d{2})(\d{2})\.\d+', paper.get("id", ""))
                match_old = re.match(r'([a-z\-]+)/(\d{2})(\d{2})\d+', paper.get("id", ""))
                if match_new:
                    yy, mm = match_new.groups()
                    year = int(yy)
                    if year < 91:
                        year += 2000
                if match_old:
                    yy, mm = match_old.groups()[1:3]
                    year = int(yy)
                    if year >= 91:
                        year += 1900

            # year = journal_ref[-4:]
            records.append({
                'id': paper['id'],
                'authors': paper['authors'],
                'title': paper['title'],
                'abstract': paper['abstract'],
                'categories': paper['categories'],
                "year": year,
            })
        return records


def filter_and_clean_df(records):
    df = pd.DataFrame(records)

    df.fillna('unknown', inplace=True)

    ai_categories = [
        "cs.AI",  # Artificial Intelligence
        "cs.CL",  # Computation and Language (NLP)
        "cs.CV",  # Computer Vision
        "cs.LG",  # Machine Learning
        "cs.MA",  # Multiagent Systems
    ]

    df = df[df['categories'].str.contains('|'.join(ai_categories), )]
    df.to_csv('Arxiv_ai_updated.csv', index=False, index_label=False)

    docs = load_data('arxiv_ai_updated.csv')
    vector_store = pgvector(connection=os.getenv('CONNECTION_NAME'),
                            collection_name=os.getenv('ABSTRACT_COLLECTION'))
    print(docs[0])
    batch_size = 500
    for i in tqdm(range(0, len(docs), batch_size)):
        _ = vector_store.add_documents(documents=docs[i: i + batch_size])


def main():

    ap = argparse.ArgumentParser()
    ap.add_argument("file",
                    help="path to json file")
    args = ap.parse_args()
    path = args.file
    if os.path.exists(path):
        print('Processing file')
        records = convert_to_df(path)
        filter_and_clean_df(records)
    else:
        print('File not found')


if __name__ == '__main__':
    main()
