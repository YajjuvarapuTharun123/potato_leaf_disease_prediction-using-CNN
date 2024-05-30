from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)
model = tf.keras.models.load_model('C:/Users/91630/OneDrive/potato disease/potato_leaf_disease_model.h5')
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        img = image.load_img(filepath, target_size=(150, 150))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0
        
        prediction = model.predict(img_array)
        class_names = ['Early_Blight', 'Healthy', 'late blight']
        predicted_class = class_names[np.argmax(prediction[0])]
        
        return render_template('result.html', prediction=predicted_class, filename=filename)

if __name__ == '__main__':
    app.run(debug=True)