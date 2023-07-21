import numpy as np


def evaluate_ideal_eigenvalue():
    delta_time = 0.1
    AA = np.eye(2)
    AA[0, 1] = delta_time
    AA[1, 1] = delta_time

    print("Transition matrix A")
    print(AA)

    U, S, Vh = np.linalg.svd(AA)
    print("Singular values S", S)
    print("Eigenvectors U")
    print(U)


def evaluate_eigenvalues(
    it_max=100,
    delta_time=0.03,
    visualize=False,
    save_figure=False,
):
    lambda_max = 1.0 / delta_time

    lambda_DS = 0.8 * lambda_max
    lambda_perp = 0.1 * lambda_max
    lambda_obs = 1.0 * lambda_max


if (__name__) == "__main__":
    evaluate_ideal_eigenvalue()
