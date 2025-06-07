

import requests
import sys
from config import test_ip

params = {'query':'大模型','db_path':'/usr/src/.lancedb','table_name':'t1'}
r = requests.get(f'http://{test_ip}:5003/query_documents', params=params)
if r:
    print(r.text)

