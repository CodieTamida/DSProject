import numpy as np
from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))


@app.route('/api/', methods = ['POST'])
def predict():
    data = request.get_json(force = True)
    prediction = model.predict([[np.array(data['fa'], data['va'], data['ca'], data['rs'],
                                          data['chlorides'], data['fsd'], data['tsd'],
                                           data['density'], data['ph'], data['sulphates'],
                                            data['alcohol'])]])
    output = prediction[0]
    return jsonify(output)


if __name__ == '__main__':
    try:
        app.run(port=5000, debug = True)
    except:
        print("Server is exited unexpectedly. Please contact server admin.")

