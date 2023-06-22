import numpy as np
import pytest

from analytics.q_matrix_calculation import calc_s_0_matrix, calculate_q_0_1_matrix


@pytest.mark.parametrize('s_matrix,result', [
    (
            np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
            np.array([[6], [15], [24]])
    )
])
def test_s_0_matrix_calculation(s_matrix, result):
    s_0_matrix = calc_s_0_matrix(s_matrix)

    # 3 rows and 1 columns
    assert np.array_equal(result, s_0_matrix)

