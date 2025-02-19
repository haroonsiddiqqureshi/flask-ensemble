import onnxruntime as ort
import numpy as np
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

onnx_model_path = "./ensemble_model.onnx"
ort_session = ort.InferenceSession(onnx_model_path)

app = Flask(__name__)
CORS(app)


@app.route("/")
def home():
    return render_template("index.html"), 200


@app.route("/predict", methods=["POST"])
def predict():
    precipitation = request.form["precipitation"]
    temp_max = request.form["temp_max"]
    temp_min = request.form["temp_min"]
    wind = request.form["wind"]

    input_data = (float(precipitation), float(temp_max), float(temp_min), float(wind))
    input_array = np.array([input_data], dtype=np.float32)

    inputs = {ort_session.get_inputs()[0].name: input_array.reshape(1, -1)}
    prediction = ort_session.run(None, inputs)

    predict = prediction[0][0]
    return predict


@app.route("/predict/line", methods=["POST"])
def predict_line():
    try:
        input_data = request.get_json()
        input_array = np.array([list(input_data.values())], dtype=np.float32)

        inputs = {ort_session.get_inputs()[0].name: input_array.reshape(1, -1)}
        prediction = ort_session.run(None, inputs)

        return (
            jsonify(
                {
                    "message": "Success",
                    "status": 200,
                    "prediction": prediction[0].tolist(),
                }
            ),
            200,
        )

    except Exception as e:
        return (
            jsonify({"message": "An error occurred", "error": str(e), "status": 500}),
            500,
        )


if __name__ == "__main__":
    app.run(debug=True)
