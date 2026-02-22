
import pandas as pd

df = pd.read_csv('data/origin_dataset.csv')
for i, col in enumerate(df.columns):
    print(f"{i}: {col}")
