import requests


if __name__ == '__main__':
    url = "http://localhost:9696/predict"
    client = client = {"job": "retired", "duration": 445, "poutcome": "success"}
    result = requests.post(url, json=client).json()
    print(result)
