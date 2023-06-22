import typing

import numpy as np
import pytest

from analytics.q_matrix_calculation import calculate_q_0_0_matrix


@pytest.mark.parametrize('buffer_size,d_matrices,ph2', [
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
                [1, 2, 3],
                [4, 5, 6],
                [7, 8, 9],
            ]),
            np.array([1, 2, 3])
        )
    )
])
def test(buffer_size: int,
         d_matrices: tuple[np.ndarray, np.ndarray],
         ph2: tuple[np.ndarray, np.ndarray]
         ):
    result = calculate_q_0_0_matrix(buffer_size, d_matrices, ph2)
    w = d_matrices[0].shape[0]
    ph2_size = ph2[0].shape[0]

    # size of matrices assertion
    assert result.shape == (
        w*(1 + buffer_size*ph2_size),
        w*(1 + buffer_size*ph2_size)
    )

    # first diag block assertion
    assert np.array_equal(
        d_matrices[0],
        result[0:d_matrices[0].shape[0], 0:d_matrices[0].shape[1]],
    )

