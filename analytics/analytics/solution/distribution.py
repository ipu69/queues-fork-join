import numpy as np


def resolve_matrix_square_equation(a: np.ndarray, b: np.ndarray, c: np.ndarray, eps=1e-10) -> np.ndarray:
    assert a.shape == b.shape
    assert b.shape == c.shape

    iterations = 0

    prev = 0.1 * np.eye(a.shape[0])
    current = np.ones(a.shape)

    inverse_b_matrix = np.linalg.inv(b)

    while np.sum(np.abs(current - prev)) >= eps:
        iterations += 1
        current = -np.dot(c + np.dot(np.linalg.matrix_power(prev, 2), a), inverse_b_matrix)
        prev, current = current, prev

    return prev


def find_r_matrix(q_2: np.ndarray, q_1: np.ndarray, q_0: np.ndarray) -> np.ndarray:
    g_matrix = find_g_matrix(q_2, q_1, q_0)

    div = np.max(np.abs(-q_1 - np.dot(q_2, g_matrix)))

    r_matrix = np.dot(
        q_2,
        np.linalg.inv((-q_1 - np.dot(q_2, g_matrix)) / div) / div
    )

    return r_matrix


def find_g_matrix(a: np.ndarray, b: np.ndarray, c: np.ndarray, eps=1e-12) -> np.ndarray:
    assert a.shape == b.shape
    assert b.shape == c.shape

    iterations = 0

    prev = np.eye(a.shape[0])
    current = np.zeros(a.shape)

    inverse_b_matrix = np.linalg.inv(-b)

    while max(np.sum(np.abs(current - prev), axis=1)) >= eps:
        iterations += 1
        current = np.dot(inverse_b_matrix, c + np.dot(a, np.linalg.matrix_power(prev, 2)))
        prev, current = current, prev

    return prev


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
        np.ones(q_0_0.shape[0])
    ))

    p_1_matrix = np.vstack((
        q_1_0.transpose(),
        (q_1 + np.dot(r, q_2)).transpose(),
        np.sum(np.linalg.inv(np.eye(r.shape[0]) - r), axis=1)
    ))

    equation_matrix = np.hstack((p_0_matrix, p_1_matrix))

    solution: np.ndarray = np.linalg.solve(
        equation_matrix[1:],
        np.vstack(
            (np.zeros((equation_matrix.shape[1] - 1, 1)), np.array([1]))
        ).flatten()
    )

    return solution[0:q_0_0.shape[0]].flatten(), solution[q_0_0.shape[0]:].flatten()


def calculate_p_i_vector(p1_vector: np.ndarray, r: np.ndarray, i: int) -> np.ndarray:
    return np.dot(p1_vector, np.linalg.matrix_power(r, i-1))


def calculate_p_i_by_vector(pi_vector: np.ndarray) -> float:
    return np.sum(pi_vector)


def calculate_p_i(p1_vector: np.ndarray, r: np.ndarray, i: int) -> float:
    return calculate_p_i_by_vector(
        calculate_p_i_vector(p1_vector, r, i)
    )


def calculate_p_i_vectors(p1_vector: np.ndarray, r: np.ndarray, max_val=10000) -> list[np.ndarray]:
    return [calculate_p_i_vector(p1_vector, r, i) for i in range(2, max_val + 1)]


def calculate_p_i_distribution(p_vectors: list[np.ndarray]) -> list:
    return [p.sum() for p in p_vectors]


def calculate_p_0_0():
    pass


def calculate_p_0_j():
    pass


def calculate_p_i_0():
    pass


def calculate_p_i_j():
    pass
