import requests
import sys
params = {'file_path':'C:\\tmp\\test_rag_doc\\several_docs\\2.docx','db_path':'/usr/src/.lancedb','table_name':'several_docs2'}
r = requests.get(f'http://localhost:5003/vectorize', params=params)
if r:
    print(r.text)