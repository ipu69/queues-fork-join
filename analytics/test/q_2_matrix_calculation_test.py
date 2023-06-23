import numpy as np
import pytest

from analytics.q_matrix_calculation import calc_s_0_matrix, calculate_q_2_matrix
from analytics.math_helper import kron


@pytest.mark.parametrize('buffer_size,ph1_size,d1_matrix,beta_2_vector', [
    (
        3,
        3,
        np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ]),
        np.array([1, 2])
    ),
    (
        0,
        12,
        np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ]),
        np.array([1, 2])
    ),
])
def test(buffer_size: int,
         ph1_size: int,
         d1_matrix: np.ndarray,
         beta_2_vector: np.ndarray, ):
    result = calculate_q_2_matrix(
        buffer_size,
        ph1_size,
        d1_matrix,
        beta_2_vector
    )

    map_size = d1_matrix.shape[0]
    ph2_size = beta_2_vector.shape[0]

    assert result.shape[0] == result.shape[1]

    if buffer_size == 0:
        assert np.array_equal(result, np.zeros((ph1_size * map_size, ph1_size * map_size)))
        return

    assert np.array_equal(
        kron(d1_matrix, np.eye(ph1_size), beta_2_vector),
        result[
                0:map_size * ph1_size,
                map_size * ph1_size:map_size * ph1_size + map_size * ph1_size * ph2_size
        ]
    )

    for i in range(buffer_size - 1):
        np.array_equal(
            kron(d1_matrix, np.eye(ph1_size*ph2_size)),
            result[
                map_size * ph1_size + i*ph1_size*ph2_size*map_size:map_size * ph1_size + (i + 1)*ph1_size*ph2_size*map_size,
                map_size * ph1_size*(1 + ph2_size) + i * ph1_size * ph2_size * map_size:map_size * ph1_size*(1 + ph2_size) + (i + 1) * ph1_size * ph2_size * map_size,
            ]
        )
