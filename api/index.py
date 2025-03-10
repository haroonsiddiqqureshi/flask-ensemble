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

    return render_template("predict.html", predict=prediction[0]), 200


@app.route("/predict/line", methods=["POST"])
def predict_line():
    try:
        data = request.get_json()
        cap_surface = data["cap-surface"]
        odor = data["odor"]
        gill_spacing = data["gill-spacing"]
        gill_size = data["gill-size"]
        stalk_root = data["stalk-root"]
        stalk_surface_below_ring = data["stalk-surface-below-ring"]
        spore_print_color = data["spore-print-color"]
        population = data["population"]
        habitat = data["habitat"]

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
            message = "เห็ดชนิดนี้ปลอยภัย✔️ กินได้🍴"
        elif prediction[0] == 1:
            message = "เห็ดชนิดนี้มีพิษ☠️ กินไม่ได้❌"

        return (
            jsonify(
                {
                    "message": "Success",
                    "status": 200,
                    "prediction": message,
                }
            ),
            200,
        )

    except Exception as e:
        print(e)
        return (
            jsonify({"message": "An error occurred", "error": str(e), "status": 500}),
            500,
        )


if __name__ == "__main__":
    app.run(debug=True)
