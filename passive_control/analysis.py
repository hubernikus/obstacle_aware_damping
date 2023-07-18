import numpy as np
import pandas as pd


def clean_array(Serie: pd.Series):
    clean = Serie.copy()
    for i in range(Serie.shape[0]):
        yy = Serie[i]
        yy = yy.replace("[", " ")
        yy = yy.replace("]", " ")
        yy = yy.replace("\n", " ")
        yy = np.array(yy.split()).astype(float)
        clean[i] = yy
    return clean


def clean_matrix(Serie: pd.Series):
    clean = Serie.copy()
    for i in range(Serie.shape[0]):
        yy = Serie[i]
        yy = yy.replace("[", " ")
        yy = yy.replace("]", " ")
        yy = yy.replace("\n", " ")
        yy = np.array(yy.split()).astype(float)
        yy = yy.reshape(3, 3)
        clean[i] = yy
    return clean


def clean_float(Serie: pd.Series):
    clean = Serie.copy()
    for i in range(Serie.shape[0]):
        yy = Serie[i]
        yy = yy.replace("[", " ")
        yy = yy.replace("]", " ")
        yy = yy.replace("\n", " ")
        yy = float(yy)
        clean[i] = yy
    return clean


def clean_all(df):
    clean_D = clean_matrix(df["D"])
    # clean_eigen_values_obs = clean_matrix(df["eigen_values_obs"]).apply(np.diag)
    clean_pos = clean_array(df["pos"])
    clean_vel = clean_array(df["vel"])
    clean_des_vel = clean_array(df["des_vel"])
    clean_normal = clean_array(df["normals"])
    clean_dist = clean_float(df["dists"])
    df_clean = pd.concat(
        [
            clean_D,
            # clean_eigen_values_obs,
            clean_vel,
            clean_des_vel,
            clean_pos,
            clean_normal,
            clean_dist,
        ],
        axis=1,
    )
    return df_clean
