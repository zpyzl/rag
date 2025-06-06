

import requests
import sys
params = {'db_path':'/usr/src/.lancedb','table_name':'t1'}
r = requests.get(f'http://123.60.191.71:5003/create_table', params=params)
print(r)
if r:
    print(r.text)

