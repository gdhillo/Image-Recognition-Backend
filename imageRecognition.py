from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import keras
from keras.applications import ResNet50
from keras.applications import InceptionV3
from keras.applications import MobileNetV2
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils

app = Flask(__name__)
model = None
graph = None

def load_model():
    global model
    model = ResNet50(weights="imagenet")
    global graph
    graph = tf.get_default_graph()


def prepare_image(image, target):
    image = image.resize(target)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = imagenet_utils.preprocess_input(image)
    return image


@app.route("/predictImage", methods=["POST"])
def predict():
    data = {"success": False}
    if request.method == "POST":
        if request.files.get("image"):
            image = request.files["image"].read()
            image = Image.open(io.BytesIO(image))
            image = prepare_image(image, target=(224, 224))
            with graph.as_default():
                preds = model.predict(image)
            results = imagenet_utils.decode_predictions(preds)
            data["predictions"] = []
            for (imagenetID, label, prob) in results[0]:
                r = {"label": label, "probability": float(prob)}
                data["predictions"].append(r)
            data["success"] = True
    return jsonify(data)

def main():
    load_model()
    app.run(debug=True)

if __name__ == "__main__":
    main()
