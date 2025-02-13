import os
import pickle
import boto3
from flask import Flask, request, jsonify
from pydantic import BaseModel, ValidationError
from flask_cors import CORS
from io import BytesIO
from dotenv import load_dotenv

load_dotenv()

hostname = os.getenv("VULTR_S3_HOSTNAME")
secret_key = os.getenv("VULTR_S3_SECRET")
access_key = os.getenv("VULTR_S3_ACCESS")
bucket_name = os.getenv("VULTR_S3_BUCKET_NAME")
model_key = 'ensemble_model.pkl'

app = Flask(__name__)

CORS(app)

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
    model_data = obj['Body'].read()
    model = pickle.load(BytesIO(model_data))
    return model

model = load_model_from_s3()

@app.route('/predict',methods=['POST'])
def predict():
    return 'Predicted'

if __name__ == "__main__":
    app.run(debug=True)