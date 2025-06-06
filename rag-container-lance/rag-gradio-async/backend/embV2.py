import requests
import json


def emb(text_list):
    url = "https://qianfan.baidubce.com/v2/embeddings"

    payload = json.dumps({
        "model": "tao-8k",
        "input": text_list
    }, ensure_ascii=False)
    headers = {
        'Content-Type': 'application/json',
        'appid': '',
        'Authorization': 'Bearer bce-v3/ALTAK-AM7Z8X6rEcqJJcnX87r1o/206e50a612994d34b965fe548bd593550d07457f'
    }

    response = requests.request("POST", url, headers=headers, data=payload.encode("utf-8"))

    print(response.text)
    return [data['embedding'] for data in json.loads(response.text)['data']]

if __name__ == '__main__':
    emb(["你好"])
