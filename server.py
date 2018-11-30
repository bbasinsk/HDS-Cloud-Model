from flask import Flask, request
import json

from classifier import get_classifier

# initialize the flask app
app = Flask(__name__)

# build the classifier
scaler, classifier = get_classifier()


@app.route('/')
def hello_world():
    return "Hello World!"


@app.route('/predict')
def make_prediction():
    if not request.get_json():
        abort(400)
    request_data = request.get_json()

    s_l = float(request_data.get("sepal_length"))
    s_w = float(request_data.get("sepal_width"))
    p_l = float(request_data.get("petal_length"))
    p_w = float(request_data.get("petal_width"))

    x = [[s_l, s_w, p_l, p_w]]
    x_transformed = scaler.transform(x)
    result = classifier.predict(x_transformed)
    return str(result[0])

if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0')
