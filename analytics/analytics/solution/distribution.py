import numpy as np
import scipy

def resolve_matrix_square_equation(a: np.ndarray, b: np.ndarray, c: np.ndarray, eps=1e-15) -> np.ndarray:
    assert a.shape == b.shape
    assert b.shape == c.shape

    iterations = 0

    prev = np.eye(a.shape[0])
    current = np.zeros(a.shape)

    inverse_b_matrix = np.linalg.inv(b)

    while np.sum(np.abs(current - prev)) >= eps:
        iterations += 1
        current = -np.dot(c + np.dot(np.linalg.matrix_power(prev, 2), a), inverse_b_matrix)
        prev, current = current, prev

    return prev


def find_r_matrix(q_2: np.ndarray, q_1: np.ndarray, q_0: np.ndarray) -> np.ndarray:
    g_matrix = find_g_matrix(q_2, q_1, q_0)

    print('g matrix')

    print(g_matrix)

    r_matrix = np.dot(
        q_2,
        np.linalg.inv((-q_1 - np.dot(q_2, g_matrix)))
    )

    return r_matrix


def find_g_matrix(a: np.ndarray, b: np.ndarray, c: np.ndarray, eps=1e-15) -> np.ndarray:
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
        q_0: np.ndarray,
        r: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:

    equation_matrix = np.hstack((
        np.vstack((
            np.ones((q_0_0.shape[0], 1)),
            np.sum(np.linalg.inv(np.eye(r.shape[0]) - r), axis=1).reshape((r.shape[0], 1))
        )),
        np.vstack((
            np.hstack((q_0_0, q_0_1)),
            np.hstack((q_1_0, q_1 + np.dot(r, q_0)))
        ))
    ))

    solution: np.ndarray = np.linalg.solve(
        equation_matrix.transpose()[0:-1],
        np.vstack(
            (np.array([1]), np.zeros((equation_matrix.shape[0] - 1, 1)))
        ).flatten(),
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


def calculate_p_i_vectors(p1_vector: np.ndarray, r: np.ndarray, max_val=1000) -> list[np.ndarray]:
    res = []
    p_vector = calculate_p_i_vector(p1_vector, r, 2)
    res.append(p_vector)
    i = 3

    while np.sum(p_vector) > 1e-7:
        p_vector = calculate_p_i_vector(p1_vector, r, i)
        res.append(p_vector)
        i = i + 1

    return res


def calculate_p_i_distribution(p_vectors: list[np.ndarray]) -> list:
    return [p.sum() for p in p_vectors]


def calculate_p_0_0(p_0_vector: np.ndarray, buffer_size: int, map_size: int, ph2_size: int):
    return np.dot(
        p_0_vector,
        np.concatenate((
            np.ones(map_size),
            np.zeros(map_size * buffer_size * ph2_size),
        ))
    )


def calculate_p_0_j(p_0_vector: np.ndarray, j: int, buffer_size: int, map_size: int, ph2_size: int):
    return np.dot(
        p_0_vector,
        np.concatenate((
            np.zeros(map_size * (1 + (j - 1) * ph2_size)),
            np.ones(map_size * ph2_size),
            np.zeros(map_size * (buffer_size - j) * ph2_size),
        ))
    )


def calculate_p_i_0(p_i_vector: np.ndarray, buffer_size: int, map_size: int, ph1_size: int, ph2_size: int):
    return np.dot(
        p_i_vector,
        np.concatenate((
            np.ones(map_size * ph1_size),
            np.zeros(map_size * buffer_size * ph1_size * ph2_size),
        ))
    )


def calculate_p_i_j(p_i_vector: np.ndarray, j: int, buffer_size: int, map_size: int, ph1_size: int, ph2_size: int):
    return np.dot(
        p_i_vector,
        np.concatenate((
            np.zeros(map_size * ph1_size * (1 + (j - 1) * ph2_size)),
            np.ones(map_size * ph1_size * ph2_size),
            np.zeros(map_size * ph1_size * (buffer_size - j) * ph2_size),
        ))
    )


def calculate_p_i_j_distribution(p_vectors: list[np.ndarray], buffer_size: int, map_size: int, ph1_size: int, ph2_size: int) -> list:
    distribution = []# array of arrays

    for index, pi_vector in enumerate(p_vectors):
        distribution.append([])

        for j in range(0, buffer_size + 1):

            if index == 0:
                if j == 0:
                    current = calculate_p_0_0(pi_vector, buffer_size, map_size, ph2_size)
                else:
                    current = calculate_p_0_j(pi_vector, j, buffer_size, map_size, ph2_size)
            else:
                if j == 0:
                    current = calculate_p_i_0(pi_vector, buffer_size, map_size, ph1_size, ph2_size)
                else:
                    current = calculate_p_i_j(pi_vector, j, buffer_size, map_size, ph1_size, ph2_size)

            distribution[index].append(current)

    return distribution


def calculate_q_j_distribution(p_vectors: list[np.ndarray], buffer_size: int, map_size: int, ph1_size: int, ph2_size: int) -> list:
    p_i_j_distrib = calculate_p_i_j_distribution(
        p_vectors,
        buffer_size,
        map_size,
        ph1_size,
        ph2_size
    )

    result = []

    for j in range(0, buffer_size + 1):
        current = 0
        for p_i_j in p_i_j_distrib:
            current += p_i_j[j]

        result.append(current)
    return result