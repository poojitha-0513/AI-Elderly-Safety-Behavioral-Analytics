import pandas as pd
import joblib

model = joblib.load(
    "models/risk_model.pkl"
)

features = pd.read_csv(
    "data/features.csv"
)

prediction = model.predict(
    features
)

print(
    "\nPredicted Risk Level:"
)

print(
    prediction[0]
)