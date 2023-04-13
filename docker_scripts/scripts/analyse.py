import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def vizualize(df):
    plt.plot([df["dists"][:, 0], df["D"][0, 0]])
    plt.show()


if __name__ == "__main__":
    df = pd.read_csv("data_robot/test_hit_from_top_damping.csv")
    # vizualize(df)
