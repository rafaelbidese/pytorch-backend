from flask import Flask, request, jsonify, redirect
from model import MAL
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

model = MAL()


@app.route("/")
@app.route("/index")
def index():
    return "I am working :D"


@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        if "image" not in request.files:
            return redirect(request.url)

        image = request.files.get("image")

        if image.filename == "":
            return redirect(request.url)

        prediction = model.predict(image)
        jsonified = jsonify({"coco": prediction})
        return jsonified
        


app.run(debug=True)
