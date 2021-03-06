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
    data =request.files['file']
    filename = secure_filename(file.filename) # save file 
    filepath = os.path.join(app.config['imgdir'], filename);
    file.save(filepath)
    image = cv2.imread(filepath)
    
    
    text_from_pickle = pickle.loads(model)
    
    # Use the loaded pickled model to make predictions
    output = text_from_pickle(image)

    return render_template('index.html', prediction_text='Number Plate:{}' .format(output))


if __name__ == "__main__":
    app.run(debug=True)
