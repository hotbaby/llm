# encoding: utf8

import json
import faiss
import pickle
from langchain import OpenAI
from langchain.chains import VectorDBQAWithSourcesChain
from langchain.prompts.prompt import PromptTemplate
from langchain.embeddings import LlamaCppEmbeddings

LlamaCppEmbeddings.embed_documents


with open("faiss_store.pkl", "rb") as f:
    store = pickle.load(f)

store.index = faiss.read_index("docs.index")


prompt_template = """使用上下文来回答最后的问题。如果你不知道答案，就说你不知道，不要试图编造答案。

{context}

问题: {question}
中文答案:"""

print("initialize chain")
chain = VectorDBQAWithSourcesChain.from_llm(llm=OpenAI(temperature=0),
                                            vectorstore=store,
                                            question_prompt=PromptTemplate(template=prompt_template, 
                                                                           input_variables=["context", "question"]))


def search(query: str):
    # query = "Transformer的网络结构"
    print(f"问题: {query}")
    result = chain({"question": query}, return_only_outputs=True)
    # print(json.dumps(result, ensure_ascii=False, indent=4))
    print(f"答案: {result['answer']}")
    print(f"来源: {result['sources']}")

query_list = [
    # "详细介绍下Transformer architecture",
    # "什么是pipeline parallel?",
    # "详细介绍下Data parallel",
    "Wenet模型网络结构",
    "Wenet解码方式有哪些？",
    "介绍下megatron框架",
]

for query in query_list:
    search(query)
    print("\n")