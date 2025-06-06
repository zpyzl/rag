

import requests
import sys
params = {'query':'个人所得税退税计算主要基于综合所得年度汇算','db_path':'/usr/src/.lancedb','table_name':'several_docs'}
r = requests.get(f'http://localhost:5003/query_documents', params=params)
if r:
    print(r.text)

