# encoding: utf8

import openai
openai.api_key = ""

inputs = ["How to train LLM?"]
print(openai.Embedding.create(input=inputs, model="text-embedding-ada-002"))
