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
    cap_surface = (int(request.form["cap_surface"]),)
    odor = (int(request.form["odor"]),)
    gill_spacing = (int(request.form["gill_spacing"]),)
    gill_size = (int(request.form["gill_size"]),)
    stalk_root = (int(request.form["stalk_root"]),)
    stalk_surface_below_ring = (int(request.form["stalk_surface_below_ring"]),)
    spore_print_color = (int(request.form["spore_print_color"]),)
    population = (int(request.form["population"]),)
    habitat = (int(request.form["habitat"]),)

    input_data = (
        cap_surface,
        odor,
        gill_spacing,
        gill_size,
        stalk_root,
        stalk_surface_below_ring,
        spore_print_color,
        population,
        habitat,
    )
    input_array = np.array([input_data], dtype=np.float32)

    inputs = {ort_session.get_inputs()[0].name: input_array.reshape(1, -1)}
    prediction = ort_session.run(None, inputs)
    
    if prediction[0] == 0:
        message = "Edibles | กินได้"
    elif prediction[0] == 1:
        message = "Poisonous! | กินไม่ได้"
    else:
        message = "No Data | ไม่มีข้อมูล"

    return render_template("predict.html", predict=message), 200

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
