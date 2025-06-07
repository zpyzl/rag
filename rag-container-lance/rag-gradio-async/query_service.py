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

CORS(app, resources=r'/*')
logger = setup_log('query_service.log',True)

db = lancedb.connect(sys.argv[1])
tbl = db.open_table(sys.argv[2])
OLD_DB_SERVICE_URL = "http://localhost:4003/query_by_filename"
PORT=sys.argv[3]

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

def query_docs(query, table, org_id, person_id, secret_level):
    docs = query_list(table, query,org_id, person_id, secret_level)
    for doc in docs:
        doc['type'] = Path(doc['filepath']).suffix
    return  docs

@app.route('/get_file', methods=['GET'])
def get_file():
    filepath = request.args.get('filepath', type=str)
    if filepath.startswith("http://"):
        return jsonify({"code": 200, "msg": 'ok', "data": filepath})
    else:
        return send_file( filepath)

@app.route('/query_documents', methods=['GET','POST'])
def query_documents():
    try:
        query = request.args.get('query', type=str)
        org_id = request.args.get('org_id', type=str)
        person_id = request.args.get('person_id', type=str)
        secret_level = request.args.get('secret_level', type=str)
        logger.info(f"查询参数：{request.args}")
        docs = query_docs(query, tbl, org_id, person_id, secret_level)
        return jsonify({"code": 200, "msg": 'ok', "data": docs})
    except Exception as e:
        logger.exception(e)


@app.route('/vectorize', methods=['GET'])
def vectorize():
    try:
        file_name = request.json.get("file_name")
        file_path = request.json.get("file_path")
        org_list = request.json.get('org_list')
        person_list = request.args.get('person_list')
        secret_level = request.args.get('secret_level')
        logger.info(f"vectorize：{request.json}")
        vectorize_org_person_file(file_name, file_path, org_list, person_list, secret_level)
        return jsonify({"code": 200, "msg": 'ok'})
    except Exception as e:
        logger.exception(e)

def vectorize_org_person_file(file_name, file_path, org_list, person_list, secret_level):
    # 查询旧库是否存在。如果文件名存在，已经向量化了
    resp = requests.get(OLD_DB_SERVICE_URL, params={"file_name": file_name})  # 查询旧向量库服务
    logger.info(f"old db resp:{resp.status_code}")
    if resp.status_code == 200:
        existing_data = resp.json()['data']
        if existing_data:
            existing_data['org_list'] = org_list
            existing_data['person_list'] = person_list
            existing_data['secret_level'] = secret_level
            logger.info("add existing_data")
            tbl.add(existing_data)
        else: # 不存在，做向量化
            vectorize_file(Path(file_path), tbl, org_list, person_list, secret_level)
    else:
        raise RuntimeError("failed to query old vec db!")


def do_query_by_filename(file_name):
    data = tbl.search().where(f"filename = '{file_name}'").to_list()
    if data:
        # 去重
        unique_data = []
        seen_id = set()
        for entry in data:
            identifier = entry["text"]
            if identifier not in seen_id:
                unique_data.append(entry)
                seen_id.add(identifier)
        return unique_data
    else:
        return []


@app.route('/query_by_filename', methods=['GET'])
def query_by_filename():
    try:
        file_name = request.args.get('file_name', type=str)
        logger.info(f"文件名：{file_name}")
        docs = do_query_by_filename(file_name)
        return jsonify({"code": 200, "msg": 'ok', "data": docs})
    except Exception as e:
        logger.exception(e)


@app.route('/create_table', methods=['GET'])
def create_table():
    try:
        db_path = request.args.get('db_path', type=str)
        table_name = request.args.get('table_name', type=str)
        db = lancedb.connect(db_path)
        db.create_table(table_name, schema=schema)
        return jsonify({"code": 200, "msg": 'ok'})
    except Exception as e:
        logger.exception(e)

if __name__ == '__main__':
    app.run("0.0.0.0",port=PORT)
