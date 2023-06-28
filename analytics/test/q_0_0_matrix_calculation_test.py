import numpy as np
import pytest

from analytics.q_matrix_calculation import calculate_q_0_0_matrix, calc_s_0_matrix
from analytics.math_helper import kron, kron_sum


@pytest.mark.parametrize('buffer_size,d_matrices,ph2', [
    (
        5,
        (
            np.array([
                [1, 2],
                [3, 4]
            ]),
            np.array([
                [5, 6],
                [7, 8]
            ])
        ),
        (
            np.array([
                [1, 2, 3],
                [4, 5, 6],
                [7, 8, 9],
            ]),
            np.array([[1, 2, 3]])
        )
    ),
    (
        1,
        (
            np.array([
                [1, 2],
                [3, 4]
            ]),
            np.array([
                [5, 6],
                [7, 8]
                ])
            ),
        (
            np.array([
                [1, 2, 3],
                [4, 58, 6],
                [7, 80, 91],
            ]),
            np.array([[1, 62, 3]])
        )
    ),
])
def test(buffer_size: int,
         d_matrices: tuple[np.ndarray, np.ndarray],
         ph2: tuple[np.ndarray, np.ndarray]
         ):
    result = calculate_q_0_0_matrix(buffer_size, d_matrices, ph2)
    w = d_matrices[0].shape[0]
    ph2_size = ph2[0].shape[0]
    beta2_vector = ph2[1]

    s2_matrix = ph2[0]
    w = d_matrices[0].shape[0]

    # size of matrices assertion
    assert result.shape == (
        w*(1 + buffer_size*ph2_size),
        w*(1 + buffer_size*ph2_size)
    )

    # first diag block assertion
    assert np.array_equal(
        d_matrices[0],
        result[0:w, 0:w],
    )

    if buffer_size == 0:
        assert np.array_equal(
            d_matrices[0],
            result,
        )
        return

    if buffer_size == 1:
        assert np.array_equal(
            kron_sum(d_matrices[0] + d_matrices[1], s2_matrix),
            result[w:, w:],
        )
        return

    for i in range(buffer_size - 1):

        assert np.array_equal(
            kron_sum(d_matrices[0], s2_matrix),
            result[
                w + i * w * ph2_size:w + (i + 1) * w * ph2_size,
                w + i * w * ph2_size:w + (i + 1) * w * ph2_size,
            ],
        )

        # compare sub diag elements
        assert np.array_equal(
            kron(np.eye(w), np.dot(calc_s_0_matrix(s2_matrix), beta2_vector)),
            result[
                w + (i + 1) * w * ph2_size:w + (i + 2) * w * ph2_size,
                w + i * w * ph2_size:w + (i + 1) * w * ph2_size,
            ],
        )


