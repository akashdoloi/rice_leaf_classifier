from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import os

app = Flask(__name__)
model = load_model("rice_leaf_model.keras")
class_labels = ['Bacterial Leaf Blight', 'Brown Spot', 'Healthy Rice Leaf', 'Leaf Blast','Leaf Scald','Sheath Blight']

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    img_path = None

    if request.method == "POST":
        file = request.files["file"]
        if file:
            img = Image.open(file).convert("RGB")
            img = img.resize((224, 224))
            img_array = image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            preds = model.predict(img_array)
            predicted_class = class_labels[np.argmax(preds)]
            confidence = round(np.max(preds) * 100, 2)
            prediction = f"{predicted_class} ({confidence}%)"

            img_path = os.path.join("static", "preview.jpg")
            img.save(img_path)

    return render_template("index.html", prediction=prediction, img_path=img_path)

if __name__ == "__main__":
    app.run(debug=True)
