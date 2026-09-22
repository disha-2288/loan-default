import pandas as pd
import numpy as np
import joblib
import os
import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score

def train_and_save():
    csv_path = "Loan_default.csv"
    model_path = "loan_model.pkl"
    
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    print("Loading Loan_default.csv...")
    t0 = time.time()
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df):,} rows in {time.time() - t0:.2f}s")

    if 'LoanID' in df.columns:
        df = df.drop(columns=['LoanID'])

    X = df.drop(columns=['Default'])
    y = df['Default']

    cat_cols = ['Education', 'EmploymentType', 'MaritalStatus', 'HasMortgage', 'HasDependents', 'LoanPurpose', 'HasCoSigner']
    num_cols = ['Age', 'Income', 'LoanAmount', 'CreditScore', 'MonthsEmployed', 'NumCreditLines', 'InterestRate', 'LoanTerm', 'DTIRatio']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
        ]
    )

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=4,
            random_state=42
        ))
    ])

    print("Splitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=50000, test_size=20000, stratify=y, random_state=42
    )

    print("Training Gradient Boosting Pipeline on 50,000 records...")
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

if __name__ == "__main__":
    train_and_save()
