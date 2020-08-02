from typing import Optional

from fastapi import FastAPI

from sklearn.externals import joblib

import pandas as pd

app = FastAPI()

# Use pickle to load in the pre-trained model
model = joblib.load("diabeteseModel.pkl")


@app.get("/")
def read_root():
	return {"DIABETES PREDICTION": "FAST API"}


@app.get("/api_diabetes/")
def predict(payload:str):
	#print(payload)
	values = [float(i) for i in payload.split(',')]
	
	headers = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin','BMI', 'DiabetesPedigreeFunction', 'Age']
	
	input_variables = pd.DataFrame([values],
								columns=headers, 
								dtype=float,
								index=['input'])	
	
	# Get the model's prediction
	prediction = model.predict(input_variables)
	#print("Prediction: ", prediction)
	prediction_proba = model.predict_proba(input_variables)[0][1]
	#print("Probabilities: ", prediction_proba)

	ret = {"prediction":float(prediction),"prediction_proba":float(prediction_proba)}

	return ret
