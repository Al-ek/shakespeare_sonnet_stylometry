import pandas
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import joblib
import os

def train():

    if os.path.exists("xgb_sonnet_model.pkl"):
        os.remove("xgb_sonnet_model.pkl")

    df = pandas.read_csv("model_dataset.csv")

    X = df.drop(columns=["true_author"])
    le = LabelEncoder()
    y = le.fit_transform(df["true_author"])


    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = XGBClassifier(
        n_estimators=1000,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        eval_metric="mlogloss"
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    print("XGBoost Accuracy:", accuracy_score(y_test, predictions))
    # print(classification_report(y_test, predictions))

    joblib.dump(model, "xgb_sonnet_model.pkl")
    joblib.dump(le, "xgb_label_encoder.pkl")

    print("Model saved.")