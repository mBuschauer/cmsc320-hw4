import pandas as pd
import numpy as np
from collections import Counter, defaultdict
import datetime
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def main():
    df = pd.read_parquet('data/reviewed_professors.parquet')
    df = df.sort_values("average_rating")

    high_reviews = df[df["average_rating"] > 4.5]# ["num_reviews"]
    print(high_reviews)
    print(sum(high_reviews["num_reviews"]))
    # exit()

    low_reviews = df[df["average_rating"] < 3] # ["num_reviews"]
    # print(sum(low_reviews))

    cluster_reviews = df[(df["average_rating"] > 3) & (df["average_rating"] < 3.20)]# ["num_reviews"]
    # print(cluster_reviews)
    # print(sum(cluster_reviews))


    combined = np.concat(high_reviews["reviews"].to_numpy())
    counts = Counter(reviews["rating"] for reviews in combined)
    result = {star: counts.get(star, 0) for star in range(1, 6)}
    print(result)

    all_reviews = [
        review["review"]
        for reviews_list in high_reviews["reviews"]
        for review in reviews_list
        if "review" in review and isinstance(review["review"], str)
    ]

    # Convert to Series of lengths
    review_lengths = pd.Series([len(r) for r in all_reviews])

    mean_length = review_lengths.mean()
    std_length = review_lengths.std()

    print(f"Mean review length: {mean_length:.2f}")
    print(f"Std dev review length: {std_length:.2f}")

    classes = {}

    for _, row in high_reviews.iterrows():
        name = row["name"]
        for review in row["reviews"]:
            if "course" in review:
                itera = None if review["course"] == None else review["course"][:4]
                if itera not in classes:
                    classes[itera] = 1
                else:
                    classes[itera] += 1

    print(classes)


if __name__ == "__main__":
    main()