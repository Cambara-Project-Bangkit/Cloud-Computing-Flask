import numpy as np
from flask import Flask, jsonify, request
from PIL import Image
from tensorflow.keras.models import load_model

app = Flask(__name__)

model = load_model("model_cambara.h5", compile=False)
@app.route("/prediction", methods=["POST"])
def prediction():
    image = request.files["image"]
    if image:
        img = Image.open(image)
        if img.mode != "RGB":
            img.convert("RGB")

        img = img.resize((64,64))
        img_array = np.asarray(img)
        img_array = np.expand_dims(img_array, axis=0)
        normalized = (img_array.astype(np.float32) / 255 ) 
        
        aksara_name = 'a'
        accuracy_score = 90
        response = {
            "message":{
                "aksara": aksara_name,
                "accuracy": accuracy_score
            }
        }
        return jsonify(response),200

if __name__ == "__main__":
    app.run(port=5000, debug=True)