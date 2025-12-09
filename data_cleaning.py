import pandas as pd
import numpy as np

def main():
    df = pd.read_parquet('data/professors.parquet')
    # print(df.info)
    # print(df.head)
    df["num_reviews"] = df["reviews"].apply(lambda x: len(x) if isinstance(x, np.ndarray) else 0)
    df = df[df["num_reviews"] >= 100]
    df = df.sort_values("num_reviews", ascending=False)

    print(df.iloc[0]["reviews"][0])

    # df.to_parquet("data/reviewed_professors.parquet", index=False)

if __name__ == "__main__":
    main()