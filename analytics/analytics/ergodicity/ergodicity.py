import numpy as np
from math import factorial


def get_map_lambda(d_matrices: tuple[np.ndarray, np.ndarray]) -> float:
    d_res: np.ndarray = d_matrices[0] + d_matrices[1]

    res = np.linalg.solve(
        np.vstack((
            d_res.transpose()[1:],
            np.ones(d_res.shape[0])
        )),
        np.vstack((
            np.zeros((
                d_res.shape[0] - 1,
                1
            )),
            np.array([1])
        ))
    )

    return np.dot(
        res.flatten(),
        d_matrices[1]
    ).sum()


def get_ph_mu(ph: tuple[np.ndarray, np.ndarray]) -> float:
    return -1 / np.dot(
        ph[1].flatten(), np.linalg.inv(ph[0])
    ).sum()


def get_y_j(_lambda: float, mu: float, j: int) -> float:
    sum = 0

    for i in range(0, j+1):
        sum += ((_lambda / mu)**i)/factorial(i)

    return ((_lambda / mu)**j / factorial(j)) / sum


def verify_ergodicity(q_matrices: tuple[np.ndarray, np.ndarray, np.ndarray]) -> tuple[float, float]:
    equation_matrix = q_matrices[0] + q_matrices[1] + q_matrices[2]

    res = np.linalg.solve(
        np.vstack((
            equation_matrix.transpose()[1:],
            np.ones(equation_matrix.shape[0])
        )),
        np.vstack((
            np.zeros((
                equation_matrix.shape[0] - 1,
                1
            )),
            np.array([1])
        ))
    ).flatten()

    return (np.dot(res, q_matrices[2]).sum(), np.dot(res, q_matrices[0]).sum())

