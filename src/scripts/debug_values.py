import pandas as pd

df = pd.read_csv("data/origin_dataset.csv")

print("--- (General Trust) ---")
print(df.iloc[:, 14].value_counts(normalize=True) * 100)
