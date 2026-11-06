from pathlib import Path

import pandas as pd
import sys
import sqlite3

CURRENT_DIR = Path(__file__).resolve().parent

# 2. Point precisely to the CSV file inside the same folder
CSV_PATH = CURRENT_DIR / "ecommerceDataset.csv"


def create_db_file(df):
    with sqlite3.connect("ecommerce_items.db") as conn:
        df.to_sql(name="ecommerce", con=conn, if_exists="replace", index=False)


def main():
    # load data
    df = pd.read_csv(CSV_PATH)

    # rename columns
    col_names = list(df.columns)
    df = df.rename(columns={'Household': 'category', col_names[1]: 'description'})

    #create_db_file(df)

    # drop null values
    df.dropna(axis=0, inplace=True)

    # remove special characters and extra spaces from description
    df["description"] = df["description"].str.lower()

    # 2. Remove URLs/Links
    df["description"] = df["description"].str.replace(
        r"https?://\S+|www\.\S+", "", regex=True
    )

    # 3. Remove brackets AND the text inside them FIRST
    df["description"] = df["description"].str.replace(
        r"\[.*?\]|\(.*?\)", "", regex=True
    )

    # 4. Remove all remaining punctuation/special characters (keeping only letters and spaces)
    df["description"] = df["description"].str.replace(
        r"[^a-z\s]", "", regex=True
    )

    # 5. Fix whitespace: collapse multiple spaces into one, then strip edges
    df["description"] = df["description"].str.replace(
        r"\s+", " ", regex=True
    ).str.strip()


    return df


if __name__ == '__main__':
    sys.exit(main())