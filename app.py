from flask import Flask, jsonify, request
import numpy as np
import pickle

with open('model.pkl', 'rb') as file:
    model = pickle.load(file)
file.close()

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    input_data = np.array(data['input'])
    output = model.predict(input_data)
    response = {'output': output.tolist()}
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
