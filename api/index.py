import os
import boto3
import numpy as np
import onnxruntime as ort
import tempfile
from flask import Flask, request, jsonify
from pydantic import BaseModel, ValidationError
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()

hostname = os.getenv("VULTR_S3_HOSTNAME")
secret_key = os.getenv("VULTR_S3_SECRET")
access_key = os.getenv("VULTR_S3_ACCESS")
bucket_name = os.getenv("VULTR_S3_BUCKET_NAME")
model_key = "voting_ensemble_model.onnx"


def load_model_from_s3():
    session = boto3.session.Session()
    client = session.client(
        "s3",
        region_name=hostname.split(".")[0],
        endpoint_url="https://" + hostname,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
    )

    obj = client.get_object(Bucket=bucket_name, Key=model_key)
    model_data = obj["Body"].read()

    with tempfile.NamedTemporaryFile(delete=False) as temp_model_file:
        temp_model_file.write(model_data)
        temp_model_path = temp_model_file.name

    return ort.InferenceSession(temp_model_path)


session = load_model_from_s3()

app = Flask(__name__)

CORS(app)


class ModelSchema(BaseModel):
    gender: int
    age: float
    profession: int
    academic_pressure: int
    work_pressure: int
    cga: float
    study_satisfaction: int
    job_satisfaction: int
    sleep_duration: int
    dietary_habits: int
    degree: int
    have_suicidal_thoughts: int
    work_study_hours: int
    financial_stress: int
    family_history_mental_illness: int


def convert_to_input_array(input_data):
    column_mapping = {
        "gender": "Gender",
        "age": "Age",
        "profession": "Profession",
        "academic_pressure": "Academic Pressure",
        "work_pressure": "Work Pressure",
        "cga": "CGPA",
        "study_satisfaction": "Study Satisfaction",
        "job_satisfaction": "Job Satisfaction",
        "sleep_duration": "Sleep Duration",
        "dietary_habits": "Dietary Habits",
        "degree": "Degree",
        "have_suicidal_thoughts": "Have you ever had suicidal thoughts ?",
        "work_study_hours": "Work/Study Hours",
        "financial_stress": "Financial Stress",
        "family_history_mental_illness": "Family History of Mental Illness",
    }

    mapped_data = {column_mapping[key]: value for key, value in input_data.items()}

    input_array = np.array([list(mapped_data.values())]).astype(np.float32)

    return input_array


@app.route("/")
def home():
    return jsonify({"message": "Hello, World!"}), 200


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        model_data = ModelSchema(**data)

        input_array = convert_to_input_array(model_data.model_dump())

        inputs = {session.get_inputs()[0].name: input_array.reshape(1, -1)}
        prediction = session.run(None, inputs)

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
    except ValidationError as e:
        return (
            jsonify(
                {"message": "Validation Error", "errors": e.errors(), "status": 400}
            ),
            400,
        )
    except Exception as e:
        return (
            jsonify({"message": "An error occurred", "error": str(e), "status": 500}),
            500,
        )


if __name__ == "__main__":
    app.run(debug=True)
