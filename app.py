import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    file = request.files['image']
    # Read the image via file.stream
    image = Image.open(file.stream)
    
    text_from_pickle = pickle.loads(model)
    
    # Use the loaded pickled model to make predictions
    output = text_from_pickle(image)

    return render_template('index.html', prediction_text='Employee Salary should be $ {}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)