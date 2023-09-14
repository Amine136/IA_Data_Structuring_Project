# Import necessary libraries
import tensorflow as tf
import numpy as np
from flask import Flask, render_template, request
from keras.models import load_model

# Create a Flask web application
app = Flask(__name__)

# Define a dictionary to explain the predictions for each model
explanation = {
    'MWC_model.h5': ['Woman', 'Men'],
    'HSC_model.h5': ['Sad', 'Happy'],
    'Eye_color.h5': ['Blue', 'Brown', 'Green'],
    'makeUp_model.h5': ["with_makeup", "without_makeup"],
    'tottoo_model.h5': ["with_tattoo", "without_tattoo"]
}

# List of model names
list_of_model_names = ['MWC_model.h5', 'HSC_model.h5', 'Eye_color.h5', 'makeUp_model.h5', 'tottoo_model.h5', "age_6_pred.h5"]

# Function to predict based on an image and a specific model
def predict1(img_path, name_model):
    # Read and preprocess the image
    img = tf.io.read_file(img_path)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.resize(img, (256, 256))
    img = tf.cast(img, tf.float32) / 255.0

    # Load the model
    model = load_model("models/"+name_model)

    # Predict using the model
    x = model.predict(tf.expand_dims(img, 0), verbose=0)

    try:
        # Determine the prediction based on threshold values
        if x >= 0.7:
            return explanation[f'{name_model}'][0]
        elif x <= 0.3:
            return explanation[f'{name_model}'][1]
        else:
            return 'None'
    except:
        return explanation[f'{name_model}'][tf.argmax(x, axis=1)[0]]

# Function to preprocess an image
def preprocess_image(img_path):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_image(img, channels=1)
    img = tf.image.resize(img, (256, 256))
    img = img / 255.0
    img = tf.expand_dims(img, axis=0)
    return img

# List of race categories
race_list = ['Black', 'White', 'Asian', 'Indian']

# Function to predict using another set of models
def predict2(img_path, name_model):
    img = preprocess_image(img_path)
    model = load_model("models/"+name_model)
    prediction = model.predict(img, verbose=0)

    if name_model == "age_6_pred.h5":
        val = int(tf.round(prediction[0][0]).numpy())
        return f'[ {val-6} , {val+6} ]'

# Function to predict various attributes and characteristics
def predict(img_path):
    p = [
        {'label': 'Gender Prediction', 'value': ' '},
        {'label': 'Status Prediction', 'value': ' '},
        {'label': 'Color eye Prediction', 'value': ' '},
        {'label': 'Makeup Prediction', 'value': ' '},
        {'label': 'Tattoo Prediction', 'value': ' '},
        {'label': 'Age Prediction', 'value': ' '}
    ]

    # Iterate through the list of model names
    for i in range(len(list_of_model_names)):
        name_model = list_of_model_names[i]
        if name_model in explanation.keys():
            p[i]['value'] = f'{predict1(img_path, name_model)}'
        else:
            p[i]['value'] = f'{predict2(img_path, name_model)}'

    return p

# Define the main route for the web application
@app.route("/", methods=['GET', 'POST'])
def main():
    return render_template("index.html")

# Define a route to handle image uploads and predictions
@app.route("/submit", methods=['GET', 'POST'])
def get_output():
    if request.method == 'POST':
        img = request.files['my_image']
        img_path = "static/" + img.filename
        img.save(img_path)
        p = predict(img_path)

    return render_template("index.html",
                           predictions=p,
                           img_path=img_path)

# Run the Flask application
if __name__ == '__main__':
    app.run(debug=True)
