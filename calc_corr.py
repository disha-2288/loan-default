import pandas as pd

df = pd.read_csv("Loan_default.csv")
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

# encode categorical for correlation
for col in df.select_dtypes(exclude='number').columns:
    df[col] = df[col].astype('category').cat.codes

df = df.drop(columns=['loanid'], errors='ignore')

corr = df.corr()['default'].abs().sort_values()
print("Correlations with Default:")
print(corr)
