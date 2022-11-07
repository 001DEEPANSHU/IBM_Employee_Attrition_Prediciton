import pickle
from flask import Flask, jsonify, request

model_file = './models/pipe.bin'

with open(model_file, 'rb') as f_in:
    pipeline = pickle.load(f_in)

app = Flask('serve')

@app.route('/', methods=['GET'])
def home():
    return "Flask app is running"

@app.route('/predict', methods=['POST'])
def predict():
    input_data = request.get_json()

    y_pred = pipeline.predict_proba(input_data)[0, 1]
    employee_exited = y_pred >= 0.5

    result = {'Probability of Attrition': float(y_pred), 'Attrition': bool(exited)}

    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8098)