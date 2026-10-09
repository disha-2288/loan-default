import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from features import CAT_COLS, NUM_COLS, preprocess

def run_task5_checklist():
    csv_path = "Loan_default.csv"
    print("Loading data...")
    df = pd.read_csv(csv_path)

    # Clean names, duplicates, feature engineering, missing values, IQR outliers
    df, _ = preprocess(df, verbose=True)
    cat_cols, num_cols = CAT_COLS, NUM_COLS

    X = df.drop(columns=['default'])
    y = df['default']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', MinMaxScaler(), num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    models = {
        "Random Forest": RandomForestClassifier(random_state=42),
        "AdaBoost": AdaBoostClassifier(random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42)
    }

    print("\n" + "="*50)
    print(" 1, 2, 4 & 6. Model Evaluation, Overfitting & Comparison")
    print("="*50)
    results = []
    
    for name, model in models.items():
        pipeline = Pipeline([('preprocessor', preprocessor), ('classifier', model)])
        print(f"\nTraining {name}...")
        pipeline.fit(X_train, y_train)
        
        y_train_pred = pipeline.predict(X_train)
        y_test_pred = pipeline.predict(X_test)
        
        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)
        precision = precision_score(y_test, y_test_pred, zero_division=0)
        recall = recall_score(y_test, y_test_pred, zero_division=0)
        f1 = f1_score(y_test, y_test_pred, zero_division=0)
        auc = roc_auc_score(y_test, pipeline.predict_proba(X_test)[:, 1])
        
        results.append({
            "Model": name,
            "Train Acc": round(train_acc, 4),
            "Test Acc": round(test_acc, 4),
            "Precision": round(precision, 4),
            "Recall": round(recall, 4),
            "F1-score": round(f1, 4),
            "ROC-AUC": round(auc, 4)
        })
        
        print(f"[{name}] Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")
        if train_acc - test_acc > 0.05:
            print("  -> Warning: Potential Overfitting (Train score much higher)")
        elif train_acc < 0.6 and test_acc < 0.6:
            print("  -> Warning: Potential Underfitting (Both scores low)")
        else:
            print("  -> Good fit [OK]")

    print("\n--- Summary Classification Table ---")
    res_df = pd.DataFrame(results)
    print(res_df.to_string(index=False))

    print("\n" + "="*50)
    print(" 3. Cross-Validation (5-Fold)")
    print("="*50)
    # Only ~11.6% of loans default, so a model that always predicts "no default" already
    # scores ~88% accuracy. ROC-AUC measures how well defaulters are ranked above
    # non-defaulters, so it is used to pick the best model.
    best_model_name = res_df.sort_values("ROC-AUC", ascending=False).iloc[0]["Model"]
    print(f"Best model by ROC-AUC: {best_model_name}")
    print(f"Running 5-fold CV on {best_model_name}...")
    cv_pipeline = Pipeline([('preprocessor', preprocessor), ('classifier', models[best_model_name])])
    cv_scores = cross_val_score(cv_pipeline, X_train, y_train, cv=5, scoring='accuracy')
    print(f"CV Scores: {cv_scores}")
    print(f"CV Average: {cv_scores.mean():.4f} | CV Spread (Std): {cv_scores.std():.4f}")
    if cv_scores.std() < 0.05:
        print("-> Low spread: The model is stable across folds.")
    else:
        print("-> High spread: The model is unstable.")

    print("\n" + "="*50)
    print(" 5. Hyperparameter Tuning (GridSearchCV)")
    print("="*50)
    print(f"Tuning {best_model_name}...")
    param_grids = {
        "Random Forest": {'classifier__n_estimators': [50, 100], 'classifier__max_depth': [None, 10]},
        "AdaBoost": {'classifier__n_estimators': [50, 100], 'classifier__learning_rate': [0.5, 1.0]},
        "Gradient Boosting": {'classifier__n_estimators': [100, 200], 'classifier__max_depth': [3, 5]},
    }
    param_grid = param_grids[best_model_name]
    grid_search = GridSearchCV(cv_pipeline, param_grid, cv=3, scoring='roc_auc', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    print(f"Best Parameters Found: {grid_search.best_params_}")
    
    best_tuned = grid_search.best_estimator_
    tuned_pred = best_tuned.predict(X_test)
    tuned_acc = accuracy_score(y_test, tuned_pred)
    tuned_auc = roc_auc_score(y_test, best_tuned.predict_proba(X_test)[:, 1])
    print(f"Tuned Model Test Accuracy: {tuned_acc:.4f} | Test ROC-AUC: {tuned_auc:.4f}")

if __name__ == "__main__":
    run_task5_checklist()
