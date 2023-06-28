import numpy as np

import analytics.test_suit as test_suit

from analytics.solution.distribution import resolve_matrix_square_equation, resolve_p_0_and_p_1
from analytics import q_matrix_calculation


def test():
    d_matrices = test_suit.d_matrices
    ph = test_suit.ph
    buffer_size = test_suit.buffer_size

    q_2 = q_matrix_calculation.calculate_q_2_matrix(
        buffer_size,
        ph[0].shape[0],
        d_matrices[1],
        ph[1],
    )

    q_1 = q_matrix_calculation.calculate_q_1_matrix(
        buffer_size,
        d_matrices,
        ph,
        ph
    )

    q_0 = q_matrix_calculation.calculate_q_0_matrix(
        buffer_size,
        ph[0].shape[0],
        d_matrices[0].shape[0],
        ph,
    )

    r = resolve_matrix_square_equation(q_2, q_1, q_0)

    q_0_0 = q_matrix_calculation.calculate_q_0_0_matrix(
        buffer_size,
        d_matrices,
        ph
    )

    q_0_1 = q_matrix_calculation.calculate_q_0_1_matrix(
        buffer_size,
        d_matrices[1],
        (
            ph[1],
            ph[1],
        )
    )

    q_1_0 = q_matrix_calculation.calculate_q_1_0_matrix(
        buffer_size,
        ph[0].shape[0],
        d_matrices[0].shape[0],
        ph[0]
    )

    assert np.allclose(np.sum(
        np.hstack((
            q_1_0, q_1, q_2
        )),
        axis=1
    ), 0)

    assert np.allclose(np.sum(
        np.hstack((
            q_0, q_1, q_2
        )),
        axis=1
    ), 0)

    solution = resolve_p_0_and_p_1(
        q_0_0,
        q_1_0,
        q_0_1,
        q_1,
        q_2,
        r
    )

    assert min(solution[0]) >= 0
    assert max(solution[0]) <= 1
    assert sum(solution[0]) <= 1


    assert min(solution[1]) >= 0
    assert max(solution[1]) <= 1
    assert sum(solution[1]) <= 1

    assert sum(np.concatenate((solution[0], solution[1]))) <= 1


