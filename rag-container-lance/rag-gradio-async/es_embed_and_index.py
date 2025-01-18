import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import List

import lancedb
import numpy as np
import pyarrow as pa
import requests
from dotenv import load_dotenv
from tqdm import tqdm
from transformers import AutoConfig
from log_utils import setup_log

from FileChunk import FileChunk
from elasticsearch import Elasticsearch
from backend.es_index import EsIndex
EsIndex("","","")
# 加载 .env 文件
load_dotenv()

logging.basicConfig(level=logging.INFO)

logger = setup_log('embed_and_index.log',True)


TEI_URL= os.getenv("EMBED_URL") + "/embed"
DIRPATH = sys.argv[2]
TABLE_NAME = os.getenv("TABLE_NAME")
config = AutoConfig.from_pretrained(os.getenv("EMBED_MODEL"))
EMB_DIM = config.hidden_size
CREATE_INDEX = int(os.getenv("CREATE_INDEX"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE"))
NUM_PARTITIONS = int(os.getenv("NUM_PARTITIONS"))
NUM_SUB_VECTORS = int(os.getenv("NUM_SUB_VECTORS"))

HEADERS = {
    "Content-Type": "application/json"
}

if len(sys.argv) < 4:
    raise RuntimeError("argv1 should be a(add) or c(create table), argv2 should be path of lancedb table")

db = lancedb.connect('/usr/src/.lancedb')
schema = pa.schema(
    [
        pa.field("vector", pa.list_(pa.float32(), EMB_DIM)),
        pa.field("filename", pa.string()),
        pa.field("filepath", pa.string()),
        pa.field("text", pa.string()),
    ]
)
if sys.argv[1] == 'a':
    tbl = db.open_table(TABLE_NAME)
else:
    tbl = db.create_table(TABLE_NAME, schema=schema, mode=os.getenv("CREATE_TABLE_MODE"))

indices = [EsIndex("risen_app_fts_39854f62b3974d71ad3c1fbbeddd3562",# 在线会议
                   "entityUrl",
                   "222.75.9.188/Library/web/"),
           EsIndex("risen_app_fts_e4c6b228d56c463a91b5e33829f79bd9",#在线文电（新文电）
                 "entityUrl",
                 "222.75.9.186:81/oa"),
           EsIndex("risen_app_fts_0863383db34a4f3dbd4b5boe6ec19423",#在线文电（老文电）
                   "entityUrl",
                   "222.75.9.254:81"),
           EsIndex("risen_app_fts_d3200a694aba4217895f22f985b12a3a",#在线人事
                     "entityUrl",
                     "222.75.9.159/zxrs"),
           EsIndex("risen_app_fts_59a4b19d801d44b0a1d7907fc1f123c5",#领导文库
                   "entityUrl",
                   "222.75.9.167:81/ldwk"),
           EsIndex("risen_app_fts_dbd2a67f00d1448c94f0113b899ea1db",#在线信息
                     "entityUrl",
                     "222.75.9.159:81/zxxx"),
           EsIndex("risen_app_fts_11d347d917714b0fabdf24fd789ed84e",#在线文库（文字系统）
                   "entityUrl",
                   "222.75.9.167/wzxt")]

es = Elasticsearch('http://71.8.157.184:9200', basic_auth=('elastic', 'Risen@2023'))
page_size = 10
sort=[
        {
            "CREATE_TIME": {
                "order": "asc",
            }
        }
    ]

def process_index(esIndex: EsIndex):
    with open('.es_info', 'r') as f:
        last_create_time = f.readlines()[-1]

    finished = False
    while not finished:
        response = es.search(
            index=esIndex.index_name,
            size=page_size,
            sort=sort,
            body={
                'query': {
                    'match_all': {}
                }
            },
            search_after=last_create_time
        )

        hits = response['hits']['hits']

        if len(hits) == 0:
            finished = True
        else:
            embed_and_index(hits,esIndex)
            last_create_time = hits[-1]['sort']
            # 保存到文件
            with open('.es_info',"wa") as f:
                f.write(str(last_create_time)+"\n")

def embed_and_index(hits, esIndex: EsIndex):
    start = time.time()
    begin = False
    for j in tqdm(range(len(hits))):
        hit = hits[j]
        try:
            entity_content_json = hit['_source']['ENTITY_DATA']['ENTITY_CONTENT_JSON']
            file_content = entity_content_json['oaliFileContent']
            filename = entity_content_json['oaliFileTitle']
            url_suffix = entity_content_json['entityUrl']
            url = esIndex.url_prefix + url_suffix

            if not begin and sys.argv[3] != str(filename): # 没到断点，跳过
                continue
            elif not begin:
                begin = True
                logger.info(f"continue {hit}, file count:{j}")

            file_chunks = req_chunk(file_content)

            for j in range(int(np.ceil(len(file_chunks) / BATCH_SIZE))):
                file_chunk_batch = file_chunks[j * BATCH_SIZE:(j + 1) * BATCH_SIZE]
                payload = {
                    "inputs": [file_chunk.chunk for file_chunk in file_chunk_batch],# texts: list of str
                    "truncate": True
                }

                resp = requests.post(TEI_URL, json=payload, headers=HEADERS)
                if resp.status_code != 200:
                    raise RuntimeError(f"failed call embedding for {filename}")
                vectors = resp.json()

                data = [
                    {"vector": vec,"filename":filename, "filepath": url,  "text": file_chunk.chunk}
                    for vec, file_chunk in zip(vectors, file_chunk_batch)
                ]
                tbl.add(data=data)
        except Exception as e:
            logger.error(f"Unhandled exception for hit: {hit}", e)
            logger.exception(e)
    
    logger.info(f"Embedding and ingestion of {len(hits)} items took {time.time() - start}")

    # IVF-PQ indexing
    if CREATE_INDEX:
        tbl.create_index(vector_column_name='vector',num_partitions=NUM_PARTITIONS, num_sub_vectors=NUM_SUB_VECTORS)


def req_chunk(file_content):
    chunk_resp = requests.post("http://localhost:5002/chunk", json={'files': json.dumps(file_content)}).json()
    chunking_files = chunk_resp['data']
    if chunk_resp['code'] != 200:
        raise RuntimeError(f"failed chunking, files0: {file_content}")
    file_chunks: List[FileChunk] = []
    for file in chunking_files:
        if 'chunks' in file:
            chunks = file['chunks']
            for chunk in chunks:
                file_chunks.append(FileChunk(file['filename'], file['filepath'], chunk))
    return file_chunks


if __name__ == "__main__":
    for index in indices:
        process_index(index.index_name)

