"""Shared preprocessing & feature engineering for the Loan Default project.

Used by train_model.py, task5_checklist.py and myapp.py so that the model is
trained and served with exactly the same columns.
"""
import pandas as pd

TARGET = "default"

# Feature Selection (Noise Reduction) - 'maritalstatus', 'loanpurpose', 'loanterm'
# and 'dtiratio' are not used directly because of their near-zero correlation with
# the target. ('loanterm' is still used inside the credit_term_interaction feature.)
CAT_COLS = ["education", "employmenttype", "hasmortgage", "hasdependents", "hascosigner"]
BASE_NUM_COLS = ["age", "income", "loanamount", "creditscore", "monthsemployed",
                 "numcreditlines", "interestrate"]
ENGINEERED_COLS = ["income_to_loan", "credit_term_interaction"]
NUM_COLS = BASE_NUM_COLS + ENGINEERED_COLS


def clean_column_names(df):
    """'Loan Amount' -> 'loan_amount', 'CreditScore' -> 'creditscore'."""
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    return df


def add_engineered_features(df):
    """Feature Engineering (Task 5). Needs lower-case column names."""
    df = df.copy()
    if "income" in df.columns and "loanamount" in df.columns:
        df["income_to_loan"] = df["income"] / (df["loanamount"] + 1)
    if "creditscore" in df.columns and "loanterm" in df.columns:
        df["credit_term_interaction"] = df["creditscore"] * df["loanterm"]
    return df


def preprocess(df, remove_outliers=True, verbose=False):
    """Full cleaning pipeline from the Data Preprocessing checklist.

    Returns the cleaned DataFrame and a report dict describing every step.
    """
    report = {"rows_loaded": len(df)}

    # 1. Clean column names
    df = clean_column_names(df)

    # 2. Remove duplicates
    report["duplicates_removed"] = int(df.duplicated().sum())
    df = df.drop_duplicates()

    # 3. Drop identifier column
    if "loanid" in df.columns:
        df = df.drop(columns=["loanid"])

    # 4. Feature engineering
    df = add_engineered_features(df)

    # 5. Fill missing values (median for numeric, mode for categorical).
    #    Assign back instead of inplace=True so it also works with pandas Copy-on-Write.
    report["missing_values_filled"] = int(df.isnull().sum().sum())
    for col in NUM_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
    for col in CAT_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0])

    # 6. Detect and handle outliers (IQR Method) on the original numeric columns.
    #    The engineered ratios are skewed by design (e.g. high income vs. a small loan is
    #    a genuine, low-risk applicant), so filtering them would delete real customers.
    rows_before = len(df)
    if remove_outliers:
        for col in BASE_NUM_COLS:
            if col in df.columns:
                q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
                iqr = q3 - q1
                df = df[(df[col] >= q1 - 1.5 * iqr) & (df[col] <= q3 + 1.5 * iqr)]
    report["outliers_removed"] = rows_before - len(df)
    report["rows_final"] = len(df)

    if verbose:
        for k, v in report.items():
            print(f"  {k:<24} {v:,}")
    return df, report


def build_applicant_frame(**fields):
    """One-row DataFrame in the exact schema the saved pipeline expects."""
    row = {k.lower(): v for k, v in fields.items()}
    return add_engineered_features(pd.DataFrame([row]))
