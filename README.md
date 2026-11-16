# Student Performance Prediction Model

The goal of this project is to predict a student's performance score based on their study habits and past scores.

## Problem

Schools and students often want to know how daily habits affect final exam results. Using data on hours studied, previous scores, sleep hours, extracurricular activities, and sample papers practiced, this model estimates what a student is likely to score so weak areas can be spotted early.

## Data & Experiments

The dataset has 10,000 rows and 6 columns. During EDA, I found and dropped 127 exact duplicate rows so they wouldn't leak between the training and test sets. I also mapped `extracurricular_activities` from Yes/No to 1/0.

Before picking a final model, I ran a few quick tests on the validation set:
- Added a `score_per_hour` feature, which slightly increased RMSE from 2.0190 to 2.01902, so I dropped it.
- Tested removing `sample_question_papers_practiced`, which pushed RMSE up to 2.07, so I kept all 5 original features.
- Compared Linear Regression, Decision Tree Regressor, and Random Forest Regressor across different `max_depth` values. Linear Regression performed best with a validation RMSE of 2.0190 and a final test RMSE of 2.0750.

## Files

- `Midterm_project.ipynb`: EDA, feature tests, and model comparisons.
- `model.py`: Cleans the data, trains the Linear Regression model, and saves it to `model.bin`.
- `predict.py`: Flask API that loads `model.bin` and serves predictions at `/predict`.
- `Dockerfile`: Sets up the container to run the Flask app with Gunicorn.
- `requirements.txt`: Required Python packages.

## How to Run

Install dependencies and train the model:
```bash
pip install -r requirements.txt
python3 model.py
```

Run the Flask service locally:
```bash
python3 predict.py
```

In another terminal, test it with a sample student:
```bash
python3 test.py
```

### Run with Docker

```bash
docker build -t grade-predictor .
docker run -it --rm -p 9696:9696 grade-predictor
```

Run `python3 test.py` in a second terminal to get a prediction from the container.
