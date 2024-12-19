import asyncio
import json
import logging
import os

import gradio as gr
import lancedb
import numpy as np
import requests
from dotenv import load_dotenv
from huggingface_hub import AsyncInferenceClient

load_dotenv("../../.env")

# Setting up the logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# db
TABLE_NAME = os.getenv("TABLE_NAME")
TEXT_COLUMN = "text"
BATCH_SIZE = 8
NPROBES = 50
REFINE_FACTOR = 30

retriever = AsyncInferenceClient(model="http://39.170.17.192:9100" + "/embed")
reranker = AsyncInferenceClient(model="http://localhost:45481" + "/rerank")

db = lancedb.connect("/usr/src/.lancedb")
tbl = db.open_table("docs2")

TOP_K_RANK = int(4)
TOP_K_RETRIEVE = int(20)

async def retrieve_docs(query: str, k: int):
    """
        Retrieve top k items with RETRIEVER
        """
    resp = await retriever.post(
        json={
            "inputs": query,
            "truncate": True
        }
    )
    try:
        query_vec = json.loads(resp)[0]
    except:
        raise gr.Error(resp.decode())

    documents = tbl.search(
        query=query_vec
    ).nprobes(NPROBES).refine_factor(REFINE_FACTOR).limit(k).to_list()
    for doc in documents:
        doc['vector'] = ''
    return documents

async def retrieve(query: str, k: int) -> list[str]:
    documents = await retrieve_docs(query, k)
    documents = [doc[TEXT_COLUMN] for doc in documents]
    return documents

async def rerank(query: str, documents: list[str], k: int) -> list[str]:
    """
    Rerank items returned by RETRIEVER and return top k
    """
    scores = []
    for i in range(int(np.ceil(len(documents) / BATCH_SIZE))):
        resp = await reranker.post(
            json={
                "query": query,
                "texts": documents[i * BATCH_SIZE:(i + 1) * BATCH_SIZE],
                "truncate": True
            }
        )
        try:
            batch_scores = json.loads(resp)
            batch_scores = [s["score"] for s in batch_scores]
            scores.extend(batch_scores)
        except:
            raise gr.Error(resp.decode())
    documents = [doc for _, doc in sorted(zip(scores, documents))[-k:]]

    return documents

def ollama_gen(query, docs: list[str], if_stream: bool):
    doc_texts = "\\n".join([doc['text'] for doc in docs])

    prompt = f"""
    你是一个能回答问题的智能助理，请用下列文档来回答问题。
    如果你不知道答案，直接返回“未能根据搜索结果生成回答”。请将回答限制在{os.getenv("ANSWER_LIMIT")}字，并请保持回答简洁。
    问题：{query}
    文档：{doc_texts}
    回答：
    """
    param = {
            "model": os.getenv("LLM_MODEL"),
            "prompt": prompt,
            "stream": if_stream
        }
    response = requests.post("http://localhost:11434/api/generate", json=param)
    # rjson = json.loads(response.text)
    # return rjson["choices"][0]["message"]["content"]
    return response

def ollama_gen_print(query, docs: list[str]):
    response = ollama_gen(query, docs, True)
    if response.status_code == 200:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                print(chunk.decode('utf-8'))
    else:
        print(f"Failed to retrieve data: {response.status_code}")

def query_list(query):
    retrieved_docs = asyncio.run(retrieve_docs(query, TOP_K_RETRIEVE))
    # documents = asyncio.run(rerank(query1, retrieved_docs, TOP_K_RANK))
    # pprint(documents)
    return retrieved_docs




if __name__ == "__main__":
    # rows = tbl.search()
    # row_count = len(rows.to_list())
    # print(row_count)
    query1 = "谁作业未提交"
    docs1 = query_list(query1)
    print(docs1)
    # ollama_gen_print(query1, docs1)



