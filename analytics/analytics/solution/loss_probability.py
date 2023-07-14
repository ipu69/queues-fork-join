import numpy as np

from analytics.ergodicity.ergodicity import get_map_lambda
from analytics.math_helper import kron


def calculate_gamma_j(
        p_vectors: list[np.ndarray],
        j: int,
        buffer_size: int,
        map_size: int,
        ph1_size: int,
        ph2_size: int
) -> np.ndarray:
    result = np.dot(
        p_vectors[0],
        np.vstack((
            np.zeros((map_size, map_size * ph2_size)),
            np.zeros((map_size*(j-1)*ph2_size, map_size * ph2_size)),
            np.eye(map_size * ph2_size),
            np.zeros((map_size * (buffer_size - j) * ph2_size, map_size * ph2_size)),
        ))
    )

    for p_vector in p_vectors[1:]:
        result += np.dot(
                    p_vector,
                    np.vstack((
                        np.zeros((map_size * ph1_size, map_size * ph2_size)),
                        np.zeros((map_size * (j-1) * ph2_size * ph1_size, map_size * ph2_size)),
                        kron(np.eye(map_size), np.ones((ph1_size, 1)), np.eye(ph2_size)),
                        np.zeros((map_size * ph1_size * (buffer_size - j) * ph2_size, map_size * ph2_size)),
                    ))
                )

    return result


def calculate_loss_prob(
        d_matrices: tuple[np.ndarray, np.ndarray],
        p_vectors: list[np.ndarray],
        buffer_size: int,
        map_size: int,
        ph1_size: int,
        ph2_size: int
) -> float:
    gamma_j = calculate_gamma_j(
        p_vectors,
        buffer_size,
        buffer_size,
        map_size,
        ph1_size,
        ph2_size
    )

    lambda_ = get_map_lambda(d_matrices)

    return np.sum(np.dot(gamma_j, kron(d_matrices[1], np.eye(ph2_size)))) / lambda_
