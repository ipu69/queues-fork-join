import numpy as np
import pytest

from analytics.q_matrix_calculation import calc_s_0_matrix, calculate_q_0_1_matrix
from analytics.math_helper import kron


@pytest.mark.parametrize('buffer_size,d1_matrix,beta_vectors', [
    (
        3,
        np.array([
            [1, 2],
            [3, 4]
        ]),
        (
            np.array([1, 2]),
            np.array([3, 4])
        )
    ),
    (
        10,
        np.array([
            [50, 22, 10],
            [33, 14, 12],
            [31, 24, 12],
        ]),
        (
            np.array([11, 2, 13, 14]),
            np.array([3, 24])
        )
    ),
    (
        0,
        np.array([
            [50, 22, 10, 15],
            [33, 14, 12, 1],
            [31, 24, 12, 4],
            [3, 24, 2, 4]
        ]),
        (
            np.array([11, 2, 13]),
            np.array([3, 24, 1, 2, 1, 6])
        )
    )
])
def test(buffer_size, d1_matrix, beta_vectors):
    result = calculate_q_0_1_matrix(buffer_size=buffer_size,
                                    d1_matrix=d1_matrix,
                                    beta_vectors=beta_vectors
    )

    w = d1_matrix.shape[1]
    ph1_size = beta_vectors[0].shape[0]
    ph2_size = beta_vectors[1].shape[0]

    if buffer_size == 0:
        assert np.array_equal(result, np.zeros((w, ph1_size * w)))
        return

    # size of matrices assertion
    assert result.shape[0] == w*(1+buffer_size*ph2_size)
    assert result.shape[1] == w*ph1_size*(1+buffer_size*ph2_size)

    # verify first sub diag block
    assert np.array_equal(
        kron(d1_matrix, beta_vectors[0], beta_vectors[1]),
        result[0:w, w*ph1_size:w*ph1_size + w*ph1_size*ph2_size]
    )

    # verify other above diag blocks
    for i in range(buffer_size - 1):
        block = kron(d1_matrix, beta_vectors[0], np.eye(ph2_size))
        result_block = result[
                w + i * (w * ph2_size):w + (i + 1) * (w * ph2_size),
                w*ph1_size + w*(i + 1)*ph1_size*ph2_size:w*ph1_size + w*(i + 2)*ph1_size*ph2_size
            ]

        assert np.array_equal(
            block,
            result_block
        )

    # check that others blocks are zeros
    result[0:w, w * ph1_size:w * ph1_size + w * ph1_size * ph2_size] = 0

    for i in range(buffer_size - 1):
        result[ w + i * (w * ph2_size):w + (i + 1) * (w * ph2_size),
                w*ph1_size + w*(i + 1)*ph1_size*ph2_size:w*ph1_size + w*(i + 2)*ph1_size*ph2_size
            ] = 0

    assert np.all(result == 0)
