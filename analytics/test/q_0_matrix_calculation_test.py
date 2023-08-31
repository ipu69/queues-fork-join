import numpy as np
import pytest

from analytics.q_matrix_calculation import calculate_q_0_matrix, calc_s_0_matrix
from analytics.math_helper import kron


@pytest.mark.parametrize('buffer_size,ph2_size,map_size,ph1', [
    (
        3,
        3,
        3,
        (
            np.array([
                [1, 2],
                [3, 4]
            ]),
            np.array([[1, 2]])
        )
    ),
])
def test(buffer_size, ph2_size, map_size, ph1):
    result = calculate_q_0_matrix(
        buffer_size,
        ph2_size,
        map_size,
        ph1
    )

    assert result.shape[0] == result.shape[1]

    s1_matrix = ph1[0]
    beta_1_vector = ph1[1]
    ph1_size = s1_matrix.shape[0]

    multiplication_result = np.dot(calc_s_0_matrix(s1_matrix), beta_1_vector)

    if buffer_size == 0:
        assert np.array_equal(
            kron(np.eye(map_size), multiplication_result),
            result
        )

        return

    assert np.array_equal(
            kron(np.eye(map_size), multiplication_result),
            result[
                0:map_size * ph1_size,
                0:map_size * ph1_size,
            ]
    )

    for i in range(0, buffer_size):
        assert np.array_equal(
            kron(np.eye(map_size), multiplication_result, np.eye(ph2_size)),
            result[
                map_size * ph1_size * (1 + i * ph2_size): map_size * ph1_size * (1 + (i + 1) * ph2_size),
                map_size * ph1_size * (1 + i * ph2_size): map_size * ph1_size * (1 + (i + 1) * ph2_size),
            ]
        )


