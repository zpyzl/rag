import requests
import sys
from config import test_ip

# params = {'file_path':'C:\\tmp\\test_rag_doc\\several_docs\\2.docx','db_path':'/usr/src/.lancedb','table_name':'t1'}
# params = {'file_path':'/data/chat-test/uploads/2025/06/09/guidao.docx',"file_name":'guidao.docx',"org_list":"o1,o2","person_list":"p1,p2","secret_level":"s1"}
params = {'file_path':'D:\\rag.docx',"file_name":'rag.docx',"org_list":"o1,o2","person_list":"p1,p2","secret_level":"s1"}
r = requests.post(f'http://localhost:5003/vectorize', json=params,timeout=60000)
print(r)
if r:
    print(r.text)