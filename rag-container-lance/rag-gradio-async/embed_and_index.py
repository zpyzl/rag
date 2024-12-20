import json
import logging
import os
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

# 加载 .env 文件
load_dotenv()

logging.basicConfig(level=logging.INFO)

logger = setup_log('embed_and_index.log',True)


TEI_URL= os.getenv("EMBED_URL") + "/embed"
DIRPATH = r"D:\test_rag_doc"
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


def embed_and_index():
    db = lancedb.connect("/usr/src/.lancedb")
    schema = pa.schema(
        [
            pa.field("vector", pa.list_(pa.float32(), EMB_DIM)),
            pa.field("filename", pa.string()),
            pa.field("filepath", pa.string()),
            pa.field("text", pa.string()),
        ]
    )
    tbl = db.create_table(TABLE_NAME, schema=schema, mode=os.getenv("CREATE_TABLE_MODE"))
    start = time.time()
    files = Path(DIRPATH).rglob('*')
    file_list = list(files)
    for j in tqdm(range(len(file_list))):
        file = file_list[j]
        if file.is_file():
            try:
                file_loader_resp = requests.post("http://localhost:5000/load_file",
                                                 json={'file_path':str(file.resolve())}).json()
                loaded_files = file_loader_resp['data']
                if file_loader_resp['code'] != 200:
                    raise RuntimeError(file_loader_resp.text)

                file_chunks = req_chunk(file, loaded_files)

                for j in range(int(np.ceil(len(file_chunks) / BATCH_SIZE))):
                    file_chunk_batch = file_chunks[j * BATCH_SIZE:(j + 1) * BATCH_SIZE]
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
            except Exception as e:
                logger.error(f"Unhandled exception for file: {file}", e)
                logger.exception(e)
    
    logger.info(f"Embedding and ingestion of {len(list(files))} items took {time.time() - start}")

    # IVF-PQ indexing
    if CREATE_INDEX:
        tbl.create_index(vector_column_name='vector',num_partitions=NUM_PARTITIONS, num_sub_vectors=NUM_SUB_VECTORS)


def req_chunk(file, files):
    chunk_resp = requests.post("http://localhost:5002/chunk", json={'files': json.dumps(files)}).json()
    chunking_files = chunk_resp['data']
    if chunk_resp['code'] != 200:
        raise RuntimeError(chunk_resp.text)
    file_chunks: List[FileChunk] = []
    for file in chunking_files:
        if 'chunks' in file:
            chunks = file['chunks']
            for chunk in chunks:
                file_chunks.append(FileChunk(file['filename'], file['filepath'], chunk))
    return file_chunks


if __name__ == "__main__":
    embed_and_index()

