import os
import sys

import lancedb
import requests

from log_utils import setup_log


from flask import Flask, request, jsonify
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = Flask(__name__)
logger = setup_log('lancedb_serv.log',True)

db = lancedb.connect("/usr/src/.lancedb")
tbl = db.open_table("docs2")
load_dotenv()

NPROBES = 50
REFINE_FACTOR = 30
TOP_K_RETRIEVE=20
TEI_URL= os.getenv("EMBED_URL") + "/embed"
HEADERS = {
    "Content-Type": "application/json"
}
@app.route('/embed', methods=['GET','POST'])
def search():
    try:
        inputs = request.json.get('inputs')
        payload = {
            "inputs": inputs,  # texts: list of str
            "truncate": True
        }
        resp = requests.post(TEI_URL, json=payload, headers=HEADERS)
        if resp.status_code != 200:
            raise RuntimeError(f"failed call embedding for {file.resolve()}")
        vectors = resp.json()

        return jsonify({"code": 200, "msg": 'ok', "data": vectors})
    except Exception as e:
        logger.exception(e)

if __name__ == '__main__':
    app.run("0.0.0.0",port=5004)