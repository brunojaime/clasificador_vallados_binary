import numpy as np
from tensorflow.keras.preprocessing import image
import requests
from io import BytesIO
#from tensorflow.keras.models import load_model
from flask import Flask, request, jsonify

import json
from keras.models import model_from_json

# Cargar la configuración del modelo desde el archivo JSON
with open('config.json', 'r') as json_file:
    model_json = json_file.read()

# Crear un modelo de Keras a partir de la configuración
model = model_from_json(model_json)

# Cargar los pesos en el modelo
model.load_weights('model.weights.h5')
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

def prepare_image(url):
    response = requests.get(url)
    img = image.load_img(BytesIO(response.content), target_size=(160, 160))  # Ajusta el tamaño según el requerido por tu modelo
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Crea un batch que contiene una sola imagen
    return img_array




app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    image_url = data['url']
    img_array = prepare_image(image_url)
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    return jsonify({'predicted_class': int(predicted_class)})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')

