

import requests
import sys
params = {'query':'大模型','db_path':'/usr/src/.lancedb','table_name':'several_docs2'}
r = requests.get(f'http://localhost:5003/query_documents', params=params)
if r:
    print(r.text)

