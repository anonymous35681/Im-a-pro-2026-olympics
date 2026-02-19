from pathlib import Path

import pandas as pd


def convert_demographics():
    input_file = Path("data/raw/демография_по_регионам_2023.xlsx")
    output_file = Path("data/demographics_dataset.csv")
    
    # Read the specific sheet for Mordovia
    # Sheet name is "2.5.3." as identified
    df = pd.read_excel(input_file, sheet_name="2.5.3.", header=None)
    
    # The actual data starts from row 5 (0-indexed) where "Всего" is  # noqa: RUF003
    # Headers are spread across rows 3 and 4
    
    # Define column names
    columns = [
        "age",
        "total_both", "total_men", "total_women",
        "urban_both", "urban_men", "urban_women",
        "rural_both", "rural_men", "rural_women"
    ]
    
    # Select only the first 10 columns (ignore "Содержание")
    data_df = df.iloc[5:, :10].copy()
    data_df.columns = columns
    
    # Basic cleaning
    # Remove rows that are empty
    data_df = data_df.dropna(subset=["age"])
    
    # Convert numeric columns to int where possible, handling special characters like '–'  # noqa: RUF003
    for col in columns[1:]:
        data_df[col] = data_df[col].astype(str).str.replace("–", "0").str.replace(" ", "").str.replace("\xa0", "")  # noqa: RUF001
        data_df[col] = pd.to_numeric(data_df[col], errors="coerce").fillna(0).astype(int)
    
    # Remove redundant filler rows (like "в том числе в возрасте, лет")
    # These usually have 0 across all data columns in our processing
    data_df = data_df[~((data_df["age"].str.contains("в том числе", na=False)) & (data_df["total_both"] == 0))]
    
    # Trim whitespace from age
    data_df["age"] = data_df["age"].astype(str).str.strip()
    
    # Reset index
    data_df = data_df.reset_index(drop=True)
    
    # Save to CSV
    data_df.to_csv(output_file, index=False, encoding="utf-8")
    print(f"Extracted demographics data for Mordovia to {output_file}")
    print(f"Total rows: {len(data_df)}")

if __name__ == "__main__":
    convert_demographics()
