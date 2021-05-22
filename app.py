import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import cv2

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
    #read image file string data
    filestr = request.files['file'].read()
    #convert string data to numpy array
    npimg = numpy.fromstring(filestr, numpy.uint8)
    # convert numpy array to image
    image = cv2.imdecode(npimg, cv2.CV_LOAD_IMAGE_UNCHANGED)
    # Read the image via file.stream
    
    
    text_from_pickle = pickle.loads(model)
    
    # Use the loaded pickled model to make predictions
    output = text_from_pickle(image)

    return render_template('index.html', prediction_text='Number Plate: ' output)


if __name__ == "__main__":
    app.run(debug=True)
