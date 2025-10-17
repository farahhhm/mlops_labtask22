import pandas as pd
from sklearn.model_selection import train_test_split
import os

def main():
    # 1. Load dataset
    url = "https://raw.githubusercontent.com/Simatwa/movies-dataset/main/data/combined.csv"
    df = pd.read_csv(url)
    print("Loaded data:", df.shape)

    # 2. Rename column
    df = df.rename(columns={"genre": "label"})

    # 3. Keep only important columns
    df = df[["title", "label", "description"]]

    # 4. Drop missing or repeated rows
    df = df.dropna()
    df = df.drop_duplicates()

    # 5. Split into train and test
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    # 6. Save the files
    os.makedirs("data", exist_ok=True)
    train_df.to_csv("data/train.csv", index=False)
    test_df.to_csv("data/test.csv", index=False)

    print(" Done! Train:", train_df.shape, "Test:", test_df.shape)

if __name__ == "__main__":
    main()
