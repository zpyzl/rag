import json

from typing import List

import requests
import time
import numpy as np
import pyarrow as pa
import lancedb
import logging
import os

from tqdm import tqdm
from pathlib import Path
from transformers import AutoConfig

from dotenv import load_dotenv
from FileChunk import FileChunk

# 加载 .env 文件
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TEI_URL= os.getenv("EMBED_URL") + "/embed"
DIRPATH = r"D:\test_rag_doc\t\ofd"
TABLE_NAME = "docs"
config = AutoConfig.from_pretrained(os.getenv("EMBED_MODEL"))
EMB_DIM = config.hidden_size
CREATE_INDEX = int(os.getenv("CREATE_INDEX"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE"))
NUM_PARTITIONS = int(os.getenv("NUM_PARTITIONS"))
NUM_SUB_VECTORS = int(os.getenv("NUM_SUB_VECTORS"))

HEADERS = {
    "Content-Type": "application/json"
}


def embed_and_index():
    file_loader_resp = requests.post("http://localhost:5000/load_file", json={'dir_path':DIRPATH}).json()
    files = file_loader_resp['data']
    if file_loader_resp['code'] == 200:
        logger.info(f"Successfully read {len(files)} files")

    db = lancedb.connect("/usr/src/.lancedb")
    schema = pa.schema(
        [
            pa.field("vector", pa.list_(pa.float32(), EMB_DIM)),
            pa.field("filename", pa.string()),
            pa.field("filepath", pa.string()),
            pa.field("text", pa.string()),
        ]
    )
    tbl = db.create_table(TABLE_NAME, schema=schema, mode="overwrite")

    start = time.time()

    # 分块
    chunk_resp = requests.post("http://localhost:5002/chunk", json={'files': json.dumps(files)}).json()
    files = chunk_resp['data']
    if chunk_resp['code'] == 200:
        logger.info(f"Successfully chunk {len(files)} files")

    file_chunks: List[FileChunk] = []
    for file in files:
        if 'chunks' in file:
            chunks = file['chunks']
            for chunk in chunks:
                 file_chunks.append( FileChunk( file['filename'], file['filepath'], chunk))

    for i in tqdm(range(int(np.ceil(len(file_chunks) / BATCH_SIZE)))):
        file_chunk_batch = file_chunks[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
        payload = {
            "inputs": [file_chunk.chunk for file_chunk in file_chunk_batch],# texts: list of str
            "truncate": True
        }

        resp = requests.post(TEI_URL, json=payload, headers=HEADERS)
        if resp.status_code != 200:
            raise RuntimeError(resp.text)
        vectors = resp.json()

        data = [
            {"vector": vec,"filename":file_chunk.filename, "filepath": file_chunk.filepath,  "text": file_chunk.chunk}
            for vec, file_chunk in zip(vectors, file_chunk_batch)
        ]
        tbl.add(data=data)
    
    logger.info(f"Embedding and ingestion of {len(file_chunks)} items took {time.time() - start}")

    # IVF-PQ indexing
    if CREATE_INDEX:
        tbl.create_index(num_partitions=NUM_PARTITIONS, num_sub_vectors=NUM_SUB_VECTORS)


if __name__ == "__main__":
    embed_and_index()

