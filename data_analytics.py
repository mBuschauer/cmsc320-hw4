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

    high_reviews = df[df["average_rating"] > 4.2]["num_reviews"]
    print(sum(high_reviews))

    low_reviews = df[df["average_rating"] < 3]["num_reviews"]
    print(sum(low_reviews))

    cluster_reviews = df[(df["average_rating"] > 3) & (df["average_rating"] < 3.20)]["num_reviews"]
    print(sum(cluster_reviews))


if __name__ == "__main__":
    main()