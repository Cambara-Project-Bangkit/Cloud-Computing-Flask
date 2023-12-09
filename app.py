import numpy as np
from flask import Flask, jsonify, request
from PIL import Image
from tensorflow.keras.models import load_model

app = Flask(__name__)

# List of aksara labels used by the model
aksara_labels = ['Ba', 'Ca', 'Da','Ga', 'Ha','Ja', 'Ka','La','Ma','Na','Nga', 'Nya', 'Pa','Ra','Sa','Ta','Wa','Ya']

# Load the model_cambara model
model = load_model("model_cambara.h5", compile=False)

@app.route("/prediction", methods=["POST"])
def prediction():
    # Get the image and aksara from the request
    image = request.files["image"]
    aksara = request.form["data"]

    # Ensure the provided aksara in capitalize format for comparison
    aksara = aksara.capitalize()

    # Check if the provided aksara is in the list of aksara labels
    if aksara in aksara_labels:
        aksara_index = aksara_labels.index(aksara)

    # Check if an image is provided in the request
    if image:
        # Open and process the image
        img = Image.open(image)
        img = img.resize((64,64))
        img = img.convert('L')
        img_array = np.array(img)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Get predictions from the model
        predictions = model.predict(img_array)

        # Get the index of the predicted label with the highest confidence
        index_predicted_label = np.argmax(predictions)
        predicted_label = aksara_labels[index_predicted_label]

        # Get the confidence score for the provided aksara
        confidence_score = predictions[0][aksara_index] * 100
        formatted_confidence_score = "{:.2f}".format(confidence_score)
        
        # Check if predictions are None (unsuccessful prediction)
        if predictions is None:
            response = {
                "message": "Prediction failed. Please retake your handwriting"
            }
            return jsonify(response),400
        
        # Check if the provided aksara matches the predicted label
        if aksara.lower() == predicted_label.lower():
            response = {
                "message": "Success. The handwriting of " + aksara + " aksara is correct",
                "data": {
                    "accuracy": formatted_confidence_score,
                    "predicted_aksara": predicted_label
                }
            }
            return jsonify(response),200
        else:
            response = {
                "message": "Failed. The handwriting of " + aksara + " aksara is not correct",
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

# Run the Flask app 
if __name__ == "__main__":
    app.run(port=8080, debug=True)