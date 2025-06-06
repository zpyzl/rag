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
import argparse

from FileChunk import FileChunk

# 加载 .env 文件
load_dotenv()

logging.basicConfig(level=logging.INFO)

logger = setup_log('embed_and_index.log',True)

TEI_URL= os.getenv("EMBED_URL") + "/embed"
config = AutoConfig.from_pretrained(os.getenv("EMBED_MODEL"))
EMB_DIM = 1024 #config.hidden_size
CREATE_INDEX = int(os.getenv("CREATE_INDEX"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE"))
NUM_PARTITIONS = int(os.getenv("NUM_PARTITIONS"))
NUM_SUB_VECTORS = int(os.getenv("NUM_SUB_VECTORS"))

HEADERS = {
    "Content-Type": "application/json"
}

schema = pa.schema(
    [
        pa.field("vector", pa.list_(pa.float32(), EMB_DIM)),
        pa.field("filename", pa.string()),
        pa.field("filepath", pa.string()),
        pa.field("text", pa.string()),
    ]
)

def embed_and_index(db_path, table_name, dir_path, file_path=None, create_or_append='c', breakpoint_filename=None):
    db = lancedb.connect(db_path)
    TABLE_NAME = table_name

    if create_or_append == 'a':
        tbl = db.open_table(TABLE_NAME)
    else:
        tbl = db.create_table(TABLE_NAME, schema=schema, mode=os.getenv("CREATE_TABLE_MODE"))

    start = time.time()

    if file_path:
        vectorize_file(Path(file_path), tbl)
    else:
        files = Path(dir_path).rglob('*')
        file_list = list(files)

        begin = False if breakpoint_filename else True # 声明了断点，则一上来不开始
        for j in tqdm(range(len(file_list))):
            file = file_list[j]
            if file.is_file():
                try:
                    if not begin and breakpoint_filename != str(file): # 没到断点，跳过
                        continue
                    elif not begin:
                        begin = True
                        logger.info(f"continue {file}, file count:{j}")

                    vectorize_file(file, tbl)
                except Exception as e:
                    logger.error(f"Unhandled exception for file: {file}", e)
                    logger.exception(e)
                    continue

        logger.info(f"Embedding and ingestion of {len(list(files))} items took {time.time() - start}")

    if tbl.count_rows() > 100:
        logger.info(f"rows count:{tbl.count_rows()}, create index")
        tbl.create_index(vector_column_name='vector',num_partitions=NUM_PARTITIONS, num_sub_vectors=NUM_SUB_VECTORS)


def vectorize_file(file, tbl):
    file_loader_resp = requests.post("http://localhost:5000/load_file",
                                     json={'file_path': str(file.resolve())}).json()
    loaded_files = file_loader_resp['data']
    logger.info(f'file load resp code:{file_loader_resp["code"]}')
    if file_loader_resp['code'] != 200:
        raise Exception(f"file load error: {file_loader_resp['code']}")
    file_chunks = req_chunk(loaded_files)
    for j in range(int(np.ceil(len(file_chunks) / BATCH_SIZE))):
        file_chunk_batch = file_chunks[j * BATCH_SIZE:(j + 1) * BATCH_SIZE]
        payload = {
            "inputs": [file_chunk.chunk for file_chunk in file_chunk_batch],  # texts: list of str
            "truncate": True
        }

        resp = requests.post(TEI_URL, json=payload, headers=HEADERS)
        if resp.status_code != 200:
            raise RuntimeError(f"failed call embedding for {file.resolve()}")
        vectors = resp.json()

        data = [
            {"vector": vec, "filename": file_chunk.filename, "filepath": file_chunk.filepath, "text": file_chunk.chunk}
            for vec, file_chunk in zip(vectors, file_chunk_batch)
        ]
        # tbl.create_fts_index()
        # print(data)################# DEBUG
        tbl.add(data=data)


def req_chunk(files):
    chunk_resp = requests.post("http://localhost:5002/chunk", json={'files': json.dumps(files)}).json()
    chunking_files = chunk_resp['data']
    if chunk_resp['code'] != 200:
        raise RuntimeError(f"failed chunking, files0: {files[0].resolve()}")
    file_chunks: List[FileChunk] = []
    for file in chunking_files:
        if 'chunks' in file:
            chunks = file['chunks']
            for chunk in chunks:
                file_chunks.append(FileChunk(file['filename'], file['filepath'], chunk))
    return file_chunks


if __name__ == "__main__":
    embed_and_index(db_path='/usr/src/.lancedb',table_name='several_docs',
                    dir_path=r'C:\\tmp\\test_rag_doc\\several_docs\\doc', file_path=None,create_or_append='a')

