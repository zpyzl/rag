import json
import os
import re
import sys
from pathlib import Path
import uuid

import lancedb
import requests
from dotenv import load_dotenv
from flask import Flask, request, jsonify, send_file, Response, stream_with_context
from flask_cors import CORS

from backend.semantic_search import query_list, ollama_gen, TOP_K_RETRIEVE, get_prompt_by_docs
from log_utils import setup_log
from embed_and_index import vectorize_file, schema

load_dotenv()

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = Flask(__name__)
app.config.from_mapping(config)

CORS(app, resources=r'/*')
logger = setup_log('query_service.log',True)


def call_completions(param):
    response = requests.post("http://39.175.132.230:35191/v1/chat/completions",
                                 json=param, stream=True)
    return get_stream_response(response)

def get_stream_response(response):
    # 流式接收response，并转发
    for chunk in response.iter_content(chunk_size=1):
        yield chunk

@app.route('/v1/chat/completions', methods=['POST'])
def pure_chat_completions():
    param = request.json
    def generate():
        logger.info(f"call_completions param: {param}")
        for chunk in call_completions(param):
            yield chunk

    return Response(stream_with_context(generate()), mimetype='text/event-stream')

@app.route('/chat_and_rag', methods=['POST'])
def chat_and_rag():
    param = request.json

    if_rag = request.json.get('rag')
    if if_rag == "true":
        rag_param = param
        # 获取最后一个问题：messages中最后一个，获取content,拼接content
        messages = request.json.get('messages')
        query = messages[-1]['content']
        docs = request.json.get('docs')
        prompt = get_prompt_by_docs(docs, query)
        messages[-1]['content'] = prompt
        rag_param['messages'] = messages
        param = rag_param

    def generate():
        logger.info(f"call_completions param: {param}")
        for chunk in call_completions(param):
            yield chunk

    return Response(stream_with_context(generate()), mimetype='text/event-stream')


@app.route('/intent_recog', methods=['GET','POST'])
def intent_recog():
    try:
        query = request.args.get('question', type=str)
        p = (f"句子：{query}\n请判断上面句子是否是“找文件”的意图，"
             "如果是，请输出要找的文件名称，以{文件名}格式输出，否则请输出{false}")
        ans = ollama_gen(p,[],False)
        res = json.loads(ans.text)['response']
        if res == "{false}": # 不是找文件，是搜索
            isSearch = "true"
            keyword = query
        else: # 是找文件，返回文件名
            isSearch = "false"
            keyword = re.findall('{(.+)}', res)[0]
        return jsonify({"code": 200, "msg": 'ok', "isSearch": isSearch, "keyword":keyword})
    except Exception as e:
        logger.exception(e)

def distinct(docs):
    return list({d['filename']: d for d in docs}.values())

@app.route('/query_documents', methods=['GET','POST'])
def query_documents():
    try:
        query = request.args.get('query', type=str)
        logger.info(f"检索问题：{query}")
        tbl = get_connect_db_table()
        docs = query_unique_docs(query, tbl)
        return jsonify({"code": 200, "msg": 'ok', "data": docs})
    except Exception as e:
        logger.exception(e)

def query_unique_docs(query, table):
    docs = query_list(table, query)
    unique_docs = distinct(docs)
    while len(unique_docs) < TOP_K_RETRIEVE:
        existing_filenames = [d['filename'] for d in unique_docs]
        more_docs = query_list(query, existing_filenames)
        if not more_docs:
            break
        # 更多去重文件 = 去重（更多文件）
        more_docs = distinct(more_docs)
        # 已有去重文件+=更多去重文件
        unique_docs.extend(more_docs)
    docs = unique_docs[:TOP_K_RETRIEVE]
    for doc in docs:
        doc['type'] = Path(doc['filepath']).suffix

    # docs_id = uuid.uuid4()
    # cache.set(docs_id, docs)
    return  docs


# @app.route('/llm_answer', methods=['GET'])
# def llm_answer():
#     try:
#         query = request.args.get('query', type=str)
#         docs_id = request.args.get('docs_id', type=str)
#         docs = cache.get(docs_id)
#         cache.clear()
#         res = ollama_gen(query, docs, False)
#         return jsonify({"code": 200, "msg": 'ok', "data": json.loads(res.text)['response']})
#     except Exception as e:
#         logger.exception(e)

@app.route('/get_file', methods=['GET'])
def get_file():
    filepath = request.args.get('filepath', type=str)
    if filepath.startswith("http://"):
        return jsonify({"code": 200, "msg": 'ok', "data": filepath})
    else:
        return send_file( filepath)

@app.route('/vectorize', methods=['GET'])
def vectorize():
    try:
        file_path = request.args.get('file_path', type=str)
        tbl = get_connect_db_table()
        vectorize_file(file_path, tbl)
        return jsonify({"code": 200, "msg": 'ok'})
    except Exception as e:
        logger.exception(e)

def get_connect_db_table():
    db_path = request.args.get('db_path', type=str)
    table_name = request.args.get('table_name', type=str)
    db = lancedb.connect(db_path)
    tbl = db.create_table(table_name, schema=schema, mode="overwrite")
    return tbl


if __name__ == '__main__':
    app.run("0.0.0.0",port=5003)
