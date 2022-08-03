from flask import Flask, redirect, render_template, request, send_file, jsonify, url_for
import pandas as pd
import os

from titanic_model import deploy_model


app = Flask(__name__)

@app.route('/isAlive')
def index2():
    return "true"

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/prediction', methods=['POST'])
def prediction():
    df = process_inputs()
    proba = deploy_model(df)

    return "The probability you survive is {0:.2f}%".format(100 * proba[0, 1])



def process_inputs():
    '''
    Process input data for the model training.
    '''

    float_keys = ('Age', 'Fare')
    int_keys = ('Pclass', 'SibSp', 'Parch')
    inputs = {}

    for key, val in request.form.items():
        if key in float_keys:
            inputs[key] = float(val)
        elif key in int_keys:
            inputs[key] = int(val)
        else:
            inputs[key] = val

    return pd.DataFrame(inputs, index=[0])


if __name__ == '__main__':
    app.run(port=5000,host='127.0.0.1', debug = True)
