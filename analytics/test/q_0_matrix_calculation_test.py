import numpy as np
import pytest

from analytics.q_matrix_calculation import calculate_q_0_matrix
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
            np.array([1, 2])
        )
    )
])
def test(buffer_size, ph2_size, map_size, ph1):
    result = calculate_q_0_matrix(
        buffer_size,
        ph2_size,
        map_size,
        ph1
    )

    assert result.shape[0] == result.shape[1]
