import requests
import sys
from config import test_ip

# params = {'file_path':'C:\\tmp\\test_rag_doc\\several_docs\\2.docx','db_path':'/usr/src/.lancedb','table_name':'t1'}
params = {'file_path':'/data/test_docs/2.docx','db_path':'/usr/src/.lancedb','table_name':'t1'}
r = requests.get(f'http://{test_ip}:5003/vectorize', params=params)
print(r)
if r:
    print(r.text)