import numpy as np


def resolve_matrix_square_equation(a: np.ndarray, b: np.ndarray, c: np.ndarray, eps=0.1e-15) -> np.ndarray:
    assert a.shape == b.shape
    assert b.shape == c.shape

    iterations = 0

    prev = np.zeros(a.shape)
    current = np.ones(a.shape)

    inverse_b_matrix = np.linalg.inv(-b)

    while np.linalg.norm(current - prev) >= eps and iterations <= 10_000_000:
        iterations += 1

        current = np.dot(c + np.dot(np.dot(prev, prev), a), inverse_b_matrix)

        prev, current = current, prev

    return current


def resolve_p_0_and_p_1(
        q_0_0: np.ndarray,
        q_1_0: np.ndarray,
        q_0_1: np.ndarray,
        q_1: np.ndarray,
        q_2: np.ndarray,
        r: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    p_0_matrix = np.vstack((
        q_0_0.transpose(),
        q_0_1.transpose(),
        np.ones((1, q_0_0.shape[0]))
    ))

    p_1_matrix = np.vstack((
        q_1_0.transpose(),
        q_1.transpose() + np.dot(r, q_2).transpose(),
        np.dot(np.linalg.inv(np.eye(r.shape[0]) - r), np.ones((r.shape[0], 1))).transpose()
    ))

    equation_matrix = np.hstack((p_0_matrix, p_1_matrix))

    solution = np.linalg.solve(
        equation_matrix[1:],
        np.vstack(
            (np.zeros((equation_matrix.shape[1] - 1, 1)), np.array([[1]]))
        )
    )

    return solution[0:q_0_0.shape[0]], solution[q_0_0.shape[0]:]
