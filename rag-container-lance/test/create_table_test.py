

import requests
import sys
from config import test_ip

params = {'db_path':'/usr/src/.lancedb','table_name':'new_docs3'}
r = requests.get(f'http://{test_ip}:5003/create_table', params=params)
print(r)
if r:
    print(r.text)

