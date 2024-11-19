import asyncio
import logging
import json
from pathlib import Path

import gradio as gr
import numpy as np
import lancedb
import os

import requests
from huggingface_hub import AsyncInferenceClient

from dotenv import load_dotenv
from sympy import pprint

load_dotenv("../../.env")

# Setting up the logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# db
TABLE_NAME = os.getenv("TABLE_NAME")
TEXT_COLUMN = "text"
BATCH_SIZE = int(os.getenv("BATCH_SIZE"))
NPROBES = int(os.getenv("NPROBES"))
REFINE_FACTOR = int(os.getenv("REFINE_FACTOR"))

retriever = AsyncInferenceClient(model=os.getenv("EMBED_URL") + "/embed")
reranker = AsyncInferenceClient(model=os.getenv("RERANK_URL") + "/rerank")

db = lancedb.connect("/usr/src/.lancedb")
tbl = db.open_table(TABLE_NAME)


TOP_K_RANK = int(os.getenv("TOP_K_RANK"))
TOP_K_RETRIEVE = int(os.getenv("TOP_K_RETRIEVE"))

async def retrieve(query: str, k: int) -> list[str]:
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

def ollama_gen(query, docs: list[str]):
    doc_texts = "\\n".join([doc for doc in docs])

    prompt = f"""
    你是一个能回答问题的智能助理，请用下列文档来回答问题。
    如果你不知道答案，直接返回“未能根据搜索结果生成回答”。请将回答限制在{os.getenv("ANSWER_LIMIT")}字，并请保持回答简洁。
    问题：{query}
    文档：{doc_texts}
    回答：
    """
    param = {
            "model": "qwen2",
            "prompt": prompt
        }
    response = requests.post("http://localhost:11434/api/generate", json=param)
    if response.status_code == 200:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                print(chunk.decode('utf-8'))
    else:
        print(f"Failed to retrieve data: {response.status_code}")

if __name__ == "__main__":
    table = db.open_table(os.getenv("TABLE_NAME"))
    rows = table.search()
    row_count = len(rows.to_list())

    query1 = "表情包是如何制作的？"
    retrieved_docs = asyncio.run( retrieve(query1, TOP_K_RETRIEVE))
    documents = asyncio.run(rerank(query1, retrieved_docs, TOP_K_RANK))
    pprint(retrieved_docs)
    pprint(documents)
    print("大模型回答：")
    ollama_gen(query1, documents)