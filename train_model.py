import json
import os
import time

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

from features import CAT_COLS, NUM_COLS, TARGET, preprocess


def train_and_save():
    csv_path = "Loan_default.csv"
    model_path = "loan_model.pkl"
    meta_path = "model_meta.json"

    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    print("Loading Loan_default.csv...")
    t0 = time.time()
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df):,} rows in {time.time() - t0:.2f}s")

    # Clean names, duplicates, feature engineering, missing values, IQR outliers
    df, report = preprocess(df, verbose=True)
    print(f"Rows after preprocessing: {len(df):,}")

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", MinMaxScaler(), NUM_COLS),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS),
        ]
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=5,
            random_state=42,
        )),
    ])

    print("Splitting dataset (80/20, stratified)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    print(f"Training Gradient Boosting Pipeline on {len(X_train):,} records...")
    t1 = time.time()
    pipeline.fit(X_train, y_train)
    print(f"Training completed in {time.time() - t1:.2f}s")

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    print(f"Validation Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print(f"Validation ROC-AUC:  {auc:.4f}")

    joblib.dump(pipeline, model_path, compress=3)
    size_mb = os.path.getsize(model_path) / (1024 * 1024)
    print(f"Model successfully saved to '{model_path}' (Size: {size_mb:.2f} MB)")

    # Small model card the Streamlit app reads (no need to re-train on every visit)
    feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    importances = pipeline.named_steps["classifier"].feature_importances_
    importance = {}
    for name, value in zip(feature_names, importances):
        # 'cat__employmenttype_Unemployed' -> 'employmenttype'
        base = name.split("__", 1)[1]
        for col in CAT_COLS:
            if base.startswith(col + "_"):
                base = col
        importance[base] = importance.get(base, 0.0) + float(value)

    meta = {
        "model": "Gradient Boosting",
        "params": {"n_estimators": 200, "learning_rate": 0.05, "max_depth": 5},
        "accuracy": round(acc, 4),
        "roc_auc": round(auc, 4),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "default_rate": round(float(y.mean()), 4),
        "preprocessing": report,
        "feature_importance": dict(sorted(importance.items(), key=lambda kv: -kv[1])),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Model card saved to '{meta_path}'")


if __name__ == "__main__":
    train_and_save()
