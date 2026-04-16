from flask import Flask, request
import numpy as np
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

X = np.array([[1],[2],[3],[4]])
y = np.array([2,4,6,8])

model = LinearRegression()
model.fit(X,y)

@app.route("/")
def home():
    return "ML Model API Running"

@app.route("/predict")
def predict():
    value = float(request.args.get("x"))
    prediction = model.predict([[value]])
    return f"Prediction: {prediction[0]}"

if __name__ == "__main__":
    app.run()