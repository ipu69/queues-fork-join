import numpy as np
import pytest

from analytics.q_matrix_calculation import calc_s_0_matrix, calculate_q_1_matrix
from analytics.math_helper import kron_sum, kron


@pytest.mark.parametrize('buffer_size,d_matrices,ph1,ph2', [
    (
        3,
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
                [1, 2],
                [3, 4]
            ]),
            np.array([
                [5, 6],
            ])
        ),
        (
            np.array([
                [5, 6, 1],
                [7, 8, 1],
                [7, 1, 1]
            ]),
            np.array([
                [9, 10, 11],
            ])
        )
    ),
    (
        0,
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
                [1, 2],
                [3, 4]
            ]),
            np.array([
                [5, 6],
            ])
        ),
        (
            np.array([
                [5, 6, 1],
                [7, 8, 1],
                [7, 1, 1]
            ]),
            np.array([
                [9, 10, 11],
            ])
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
                [1, 2],
                [3, 4]
            ]),
            np.array([
                [5, 6],
            ])
        ),
        (
            np.array([
                [5, 6, 1],
                [7, 8, 1],
                [7, 1, 1]
            ]),
            np.array([
                [9, 10, 11],
            ])
        )
    )
])
def test(buffer_size: int,
         d_matrices: tuple[np.ndarray, np.ndarray],
         ph1: tuple[np.ndarray, np.ndarray],
         ph2: tuple[np.ndarray, np.ndarray]):
    result = calculate_q_1_matrix(
        buffer_size, d_matrices, ph1, ph2
    )

    map_size = d_matrices[0].shape[0]
    ph1_size = ph1[0].shape[0]
    ph2_size = ph2[0].shape[0]
    s1_matrix = ph1[0]
    s2_matrix = ph2[0]

    assert result.shape[0] == result.shape[1]
    assert result.shape[0] == map_size * ph1_size * (1 + buffer_size * ph2_size)

    if buffer_size == 0:
        assert np.array_equal(result, kron_sum(d_matrices[0], s1_matrix))
        return

    assert np.array_equal(
        result[
            0:map_size * ph1_size,
            0:map_size * ph1_size
        ],
        kron_sum(d_matrices[0], s1_matrix)
    )

    if buffer_size == 1:
        assert np.array_equal(
            result[
                map_size * ph1_size:,
                map_size * ph1_size:
            ],
            kron_sum(d_matrices[0] + d_matrices[1], s1_matrix, s2_matrix)
        )