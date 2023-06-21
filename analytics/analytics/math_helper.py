import numpy as np
import math


def kron(*args):
    """

    Args:
        *args: list of matrices, type: np.ndarray

    Returns:
        2-d matrix, np.ndarray
    """
    assert len(args) > 1

    result = args[0]
    matrices = args[1:]
    for matrix in matrices:
        result = np.kron(result, matrix)
    return result


def kron_sum(*args):
    """

    Args:
        *args: list of matrices, type: np.ndarray

    Returns:
        2-d matrix, np.ndarray
    """
    assert len(args) > 1

    result = args[0]
    matrices = args[1:]
    for matrix in matrices:
        result = np.kron(result, np.eye(matrix.shape[0])) + np.kron(np.eye(result.shape[0]), matrix)
    return result


def dot(*args):
    """

    Args:
        *args: list of matrices, type: np.ndarray

    Returns:
        2-d matrix, np.ndarray
    """
    result = args[0]
    matrices = args[1:]
    for matrix in matrices:
        result = np.dot(result, matrix)
    return result


def calc_combinations(n, k):
    assert n >= k
    return int(math.factorial(n) / (math.factorial(k) * math.factorial(n - k)))


def calc_sum_multiply_combinations(N, M1, M2, from_r, to_r):
    result = 0
    for r in range(from_r, to_r + 1):
        result += calc_combinations(r + M1 - 1, M1 - 1) * calc_combinations(N - r + M2 - 2, M2 - 1)
    return result


def calc_quantity(N, M1, M2, W, R):
    """
        Function calculates vector of possible states in ph process

        Args

            :param N - quantity of devices
            :type N: int
            :param M1 - quantity of phases in device for not priority packet
            :type M1: int
            :param M2 - quantity of phases in device for priority packet
            :type M2: int
            :param W - quantity of states in control process
            :type W: int
            :param R - max quantity of packets in queue
            :type R: int
    """

    result = np.array([(i + 1) * (W + 1) for i in range(R + 1)])
    sum_0 = 0
    for n in range(0, N + 1):
        for r in range(0, n + 1):
            sum_0 += calc_combinations(r + M1 - 1, M1 - 1) * calc_combinations(n - r + M2 - 1, M2 - 1)

    sum_res = 0
    for r in range(0, N + 1):
        sum_res += calc_combinations(r + M1 - 1, M1 - 1) * calc_combinations(N - r + M2 - 1, M2 - 1)

    result = result * sum_res

    result[0] = (W + 1) * sum_0

    return result


def calc_stat_distribution_vector(matrix):
    """

    Args:
        matrix: 2-d matrix

    Returns:
        vector is b*A=c equation solution
        where c is vector (0,0,....,0,1)
    """
    matrix_shape_x = matrix.shape[0]

    last_row = np.ones(matrix_shape_x)

    answer = np.zeros(matrix_shape_x)
    answer[-1] = 1

    return np.linalg.solve(
        np.vstack((
            matrix.transpose()[0:-1],
            last_row
        )),
        answer
    )
