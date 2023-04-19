# encoding: utf8

import json
import faiss
import pickle
from pathlib import Path, PosixPath
from json.encoder import JSONEncoder as BaseJSONEncoder

from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from pdfminer.high_level import extract_text

class JSONEncoder(BaseJSONEncoder):

    def default(self, o):
        if isinstance(o, PosixPath):
            return str(o)
        else:
            return super().default(o)


data_path = "/data/datasets/papers"
doc_json_path = "docs.json"
metadata_json_path = "metadata.json"



def parse_files(path):
    data = []
    sources = []

    paths = list(Path(path).glob("**/*.pdf"))
    for p in paths:
        data.append(extract_text(p))
        sources.append(p)

        print(f"parse pdf {p}")

    text_splitter = CharacterTextSplitter(separator="\n")
    docs = []
    metadatas = []
    for i, d in enumerate(data):
        splits = text_splitter.split_text(d)
        docs.extend(splits)
        metadatas.extend([{"source": sources[i]}] * len(splits))

    with open(doc_json_path, "w+") as f:
        f.write(json.dumps(docs, cls=JSONEncoder))

    with open(metadata_json_path, "w+") as f:
        f.write(json.dumps(metadatas, cls=JSONEncoder))


def load_docs_and_metadatas():
    with open(doc_json_path) as f:
        doc_json = json.load(f)

    with open(metadata_json_path,) as f:
        metadata_json = json.load(f)

    return doc_json, metadata_json


# parse_files(data_path)
doc_json, metadata_json = load_docs_and_metadatas()
store = FAISS.from_texts(doc_json,
                         OpenAIEmbeddings(disallowed_special=()),
                         metadatas=metadata_json)
faiss.write_index(store.index, "docs.index")
store.index = None
with open("faiss_store.pkl", "wb") as f:
    pickle.dump(store, f)
