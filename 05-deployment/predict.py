from flask import Flask
from flask import request, jsonify

from utils import read_file, predict_prob


model_file_name = 'model2.bin'
dv_file_name = 'dv.bin'

model = read_file(model_file_name)
dv = read_file(dv_file_name)

app = Flask('credit_scoring')


@app.route('/predict', methods=['POST'])
def predict():
    bank_client = request.get_json()
    credit_score = predict_prob(bank_client, model=model, dv=dv)
    result = {
        'credit_score': float(credit_score)
    }

    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)
