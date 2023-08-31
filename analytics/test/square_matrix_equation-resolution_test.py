import numpy as np

from analytics.solution.distribution import resolve_matrix_square_equation
from analytics import q_matrix_calculation

import analytics.test_suit as test_suit


def test():
    d_matrices = test_suit.d_matrices
    ph = test_suit.ph
    buffer_size = test_suit.buffer_size

    q2 = q_matrix_calculation.calculate_q_2_matrix(
        buffer_size,
        ph[0].shape[0],
        d_matrices[0],
        ph[1],
    )

    q1 = q_matrix_calculation.calculate_q_1_matrix(
        buffer_size,
        d_matrices,
        ph,
        ph
    )

    q0 = q_matrix_calculation.calculate_q_0_matrix(
        buffer_size,
        ph[0].shape[0],
        d_matrices[0].shape[0],
        ph,
    )

    solution = resolve_matrix_square_equation(q2, q1, q0)

