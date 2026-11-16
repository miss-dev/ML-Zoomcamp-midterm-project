import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error

df = pd.read_csv('Student_Performance.csv')

df.columns = df.columns.str.lower().str.replace(" ", "_")

df = df.drop_duplicates().reset_index(drop=True)

df["extracurricular_activities"] = df["extracurricular_activities"].map({"Yes": 1, "No": 0})

X = df.drop('performance_index', axis = 1)
y = df['performance_index']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

rmse = root_mean_squared_error(y_test, y_pred)
print(rmse)

output_file = "model.bin"

with open(output_file, "wb") as f_out:
    pickle.dump(model, f_out)

print(f"Saved model to {output_file}")