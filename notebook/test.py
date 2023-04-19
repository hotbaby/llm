# encoding: utf8

import openai
openai.api_key = "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"

inputs = ["How to train LLM?"]
print(openai.Embedding.create(input=inputs, model="text-embedding-ada-002"))