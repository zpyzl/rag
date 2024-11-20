import json
import os
import sys
import time

from flask import Flask, request, jsonify, send_file
from flask_caching import Cache
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

logger = setup_log('query_service.log',True)

@app.route('/query_documents', methods=['GET','POST'])
def query_documents():
    try:
        query = request.args.get('query', type=str)
        logger.info(f"检索问题：{query}")
        #docs = query_list(query)
        #cache.set("docs",docs)
        with open("r1",mode="r",encoding='utf-8') as f:
            r = f.read()
            r = json.loads(r)
        return jsonify(r)
        # return jsonify({"code": 200, "msg": 'ok', "data": r})
    except Exception as e:
        logger.exception(e)

@app.route('/llm_answer', methods=['POST'])
def llm_answer():
    try:
        query = request.json.get['query']
        docs = cache.get("docs")
        cache.clear()
        res = ollama_gen(query, docs, False)
        return jsonify({"code": 200, "msg": 'ok', "data": res.text})
    except Exception as e:
        logger.exception(e)

@app.route('/get_file', methods=['GET'])
def get_file():
    filepath = request.args.get('file_path', type=str)
    return send_file( filepath)

if __name__ == '__main__':
    app.run("0.0.0.0",port=5003)
