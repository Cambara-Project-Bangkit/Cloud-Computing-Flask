import numpy as np
from flask import Flask, jsonify, request
from PIL import Image
from tensorflow.keras.models import load_model

app = Flask(__name__)
aksara_labels = ['Ba', 'Ca', 'Da','Ga', 'Ha','Ja', 'Ka','La','Ma','Na','Nga', 'Nya', 'Pa','Ra','Sa','Ta','Wa','Ya']

model = load_model("model_cambara.h5", compile=False)
@app.route("/prediction", methods=["POST"])
def prediction():
    image = request.files["image"]
    aksara = request.form["data"]
    if image:
        img = Image.open(image)

        img = img.resize((64,64))
        img = img.convert('L')
        img_array = np.array(img)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        predictions = model.predict(img_array)

        index = np.argmax(predictions)
        predicted_label = aksara_labels[index]
        confidence_score = predictions[0][index] * 100
        formatted_confidence_score = "{:.2f}".format(confidence_score)
        
        if aksara.lower() == predicted_label.lower():
            response = {
                "message": "Success. The predicted "+ aksara.capitalize() +" aksara from the handwriting is correct",
                "data": {
                    "accuracy": formatted_confidence_score
                }
            }
            return jsonify(response),200
        else:
            response = {
                "message": "Success. The predicted "+ aksara.capitalize() +" aksara from the handwriting is not correct",
                "data": {
                    "accuracy": formatted_confidence_score,
                    "predicted_aksara": predicted_label
                }
            }
            return jsonify(response),200
    else: 
        response = {
            "message": "No image file"
        }
        return jsonify(response),400

if __name__ == "__main__":
    app.run(port=8080, debug=True)