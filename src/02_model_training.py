import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score, mean_absolute_error

pd.set_option("display.max_columns", None)

df = pd.read_csv("../data/student_performance.csv")
df = df.dropna()

print("shape:", df.shape)
print(df.head())

# features and target
x = df.drop(columns=["Performance Index"])
y = df["Performance Index"]

print("\nfeatures:", list(x.columns))
print("target range:", y.min(), "-", y.max())

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42)

# column 2 is "Extracurricular Activities" — needs one hot encoding
preprocessor = ColumnTransformer(transformers=[
    ("encode", OneHotEncoder(sparse_output=False, drop="first"), [2])
], remainder="passthrough")

# compare models
models = {
    "Linear Regression":   LinearRegression(),
    "Random Forest":       RandomForestRegressor(n_estimators=100, random_state=2),
    "Gradient Boosting":   GradientBoostingRegressor(n_estimators=100, random_state=2),
}

print("\nmodel comparison:")
print("-" * 45)
for name, model in models.items():
    pipe = Pipeline([("preprocessor", preprocessor), ("model", model)])
    pipe.fit(x_train, y_train)
    y_pred = pipe.predict(x_test)
    r2  = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"{name:25s} -> R2: {r2:.4f} | MAE: {mae:.4f}")

# best model — linear regression works very well on this dataset
best_pipe = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LinearRegression())
])
best_pipe.fit(x_train, y_train)
y_pred = best_pipe.predict(x_test)

r2  = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print("\nbest model: Linear Regression")
print(f"R2 score: {r2:.4f} ({r2*100:.2f}%)")
print(f"MAE:      {mae:.4f}")

# save model and dataframe
pickle.dump(df, open("../data/df.pkl", "wb"))
pickle.dump(best_pipe, open("../data/pipe.pkl", "wb"))
print("\nmodel saved to data/pipe.pkl")
print("dataframe saved to data/df.pkl")
