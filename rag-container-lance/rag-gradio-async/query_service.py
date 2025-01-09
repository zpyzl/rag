import json
import os
import re
import sys
import time
from pathlib import Path

from flask import Flask, request, jsonify, send_file, make_response
from flask_caching import Cache
from flask_cors import CORS
from olefile.olefile import keyword

from backend.semantic_search import query_list, ollama_gen
from log_utils import setup_log
from dotenv import load_dotenv

load_dotenv()

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
config = {
    "CACHE_TYPE": "SimpleCache",  # Flask-Caching related configs
    "CACHE_DEFAULT_TIMEOUT": 300
}

app = Flask(__name__)
app.config.from_mapping(config)
cache = Cache(app)

CORS(app, resources=r'/*')
logger = setup_log('query_service.log',True)

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


@app.route('/query_documents', methods=['GET','POST'])
def query_documents():
    try:
        query = request.args.get('query', type=str)
        logger.info(f"检索问题：{query}")
        docs = query_list(query)
        for doc in docs:
            doc['type'] = Path(doc['filepath']).suffix
        cache.set("docs",docs)
        return jsonify({"code": 200, "msg": 'ok', "data": docs})
    except Exception as e:
        logger.exception(e)

@app.route('/llm_answer', methods=['GET'])
def llm_answer():
    try:
        query = request.args.get('query', type=str)
        docs = cache.get("docs")
        cache.clear()
        res = ollama_gen(query, docs, False)
        return jsonify({"code": 200, "msg": 'ok', "data": json.loads(res.text)['response']})
    except Exception as e:
        logger.exception(e)

@app.route('/get_file', methods=['GET'])
def get_file():
    filepath = request.args.get('filepath', type=str)
    return send_file( filepath)

if __name__ == '__main__':
    app.run("0.0.0.0",port=5003)
