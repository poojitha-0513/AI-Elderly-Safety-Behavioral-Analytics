import pandas as pd
import joblib

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

df = pd.read_csv(
    "data/training_dataset.csv"
)

X = df[
    [
        "standing_time",
        "walking_time",
        "sitting_time",
        "lying_time",
        "fall_count",
        "inactivity_alerts",
        "lying_alerts",
        "movement_score"
    ]
]

y = df["risk"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

model = LogisticRegression(
    max_iter=1000
)

model.fit(
    X_train,
    y_train
)

y_pred = model.predict(
    X_test
)

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

cm = confusion_matrix(
    y_test,
    y_pred
)

report = classification_report(
    y_test,
    y_pred,
    zero_division=0
)

print("\nMODEL RESULTS\n")

print(
    f"Accuracy : {accuracy:.2f}"
)

print(
    f"Precision: {precision:.2f}"
)

print(
    f"Recall   : {recall:.2f}"
)

print(
    f"F1 Score : {f1:.2f}"
)

print("\nConfusion Matrix\n")

print(cm)

print("\nClassification Report\n")

print(report)

joblib.dump(
    model,
    "models/risk_model.pkl"
)

with open(
    "models/model_report.txt",
    "w"
) as file:

    file.write(
        "ELDERLY RISK PREDICTION MODEL\n\n"
    )

    file.write(
        f"Accuracy : {accuracy:.4f}\n"
    )

    file.write(
        f"Precision: {precision:.4f}\n"
    )

    file.write(
        f"Recall   : {recall:.4f}\n"
    )

    file.write(
        f"F1 Score : {f1:.4f}\n\n"
    )

    file.write(
        "Confusion Matrix\n"
    )

    file.write(
        str(cm)
    )

    file.write("\n\n")

    file.write(
        "Classification Report\n"
    )

    file.write(
        report
    )

print(
    "\nModel Saved Successfully"
)

print(
    "Report Saved Successfully"
)