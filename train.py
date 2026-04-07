import pandas
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

def train():

    if os.path.exists("sonnet_model.pkl"):
        os.remove("sonnet_model.pkl")

    df = pandas.read_csv("model_dataset.csv")

    X = df.drop(columns=["true_author"])
    y = df["true_author"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=5000,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, predictions))
    print(classification_report(y_test, predictions))

    joblib.dump(model, "sonnet_model.pkl")

    print("Model saved.")