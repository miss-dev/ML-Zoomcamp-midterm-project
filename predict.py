import pickle
import pandas as pd
from flask import Flask, request, jsonify

with open("model.bin", "rb") as f_in:
    model = pickle.load(f_in)

app = Flask("student-grade-predictor")

features = [
    "hours_studied",
    "previous_scores",
    "extracurricular_activities",
    "sleep_hours",
    "sample_question_papers_practiced"
]

@app.route("/predict", methods=["POST"])
def predict():
    student = request.get_json()
    
    df_input = pd.DataFrame([student])[features]
    
    prediction = model.predict(df_input)
    result = {"predicted_performance_index": round(float(prediction[0]), 2)}
    
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9696)
