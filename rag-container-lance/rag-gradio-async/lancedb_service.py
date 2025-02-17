import os
import sys

import lancedb
from log_utils import setup_log


from flask import Flask, request, jsonify

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = Flask(__name__)
logger = setup_log('lancedb_serv.log',True)

db = lancedb.connect("/usr/src/.lancedb")
tbl = db.open_table("docs2")

NPROBES = 50
REFINE_FACTOR = 30
TOP_K_RETRIEVE=20

@app.route('/search', methods=['GET','POST'])
def search():
    try:
        query_vec = request.json.get('vec')
        filenames_not_in = request.json.get('filenames_not_in')
        if filenames_not_in:
            filenames_str = "\'"
            filenames_str += "\',\'".join([filename for filename in filenames_not_in])
            filenames_str += "\'"
            documents = tbl.search(
                query=query_vec
            ).where(f"filename NOT IN ({filenames_str})").nprobes(NPROBES).refine_factor(REFINE_FACTOR).limit(
                TOP_K_RETRIEVE).to_list()
        else:
            documents = tbl.search(
                query=query_vec
            ).nprobes(NPROBES).refine_factor(REFINE_FACTOR).limit(TOP_K_RETRIEVE).to_list()

        return jsonify({"code": 200, "msg": 'ok', "data": documents})
    except Exception as e:
        logger.exception(e)

if __name__ == '__main__':
    app.run("0.0.0.0",port=5004)