import numpy as np
import pytest

from analytics.q_matrix_calculation import calc_s_0_matrix, calculate_q_1_0_matrix
from analytics.math_helper import kron


@pytest.mark.parametrize('buffer_size,ph2_size,map_size,s1_matrix', [
    (
        3,
        3,
        2,
        np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ])
    ),
    (
        0,
        4,
        7,
        np.array([
            [1, 2],
            [4, 5],
        ])
    ),
])
def test(buffer_size, ph2_size, map_size, s1_matrix):
    result = calculate_q_1_0_matrix(
        buffer_size,
        ph2_size,
        map_size,
        s1_matrix
    )

    # assert size
    assert result.shape[0] == map_size * s1_matrix.shape[0] * (1 + buffer_size * ph2_size)
    assert result.shape[1] == map_size * (1 + buffer_size * ph2_size)

    s_1_0_matrix = calc_s_0_matrix(s1_matrix)
    first_block = kron(np.eye(map_size), s_1_0_matrix)
    main_block = kron(first_block, np.eye(ph2_size))

    # assert first block in diag in matrix
    assert np.array_equal(
        first_block,
        result[0:map_size * s_1_0_matrix.shape[0], 0:map_size]
    )

    for i in range(buffer_size):
        # assert other blocks
        s_1_0_shape = s_1_0_matrix.shape[0]

        assert np.array_equal(
            main_block,
            result[
                map_size * s_1_0_shape + i * map_size * s_1_0_shape * ph2_size:map_size * s_1_0_shape + (i+1) * map_size * s_1_0_shape * ph2_size,
                map_size + i * map_size * ph2_size:map_size + (i + 1) * map_size * ph2_size,

            ]
        )

    # move to zeros all not zeros elements

    result[0:map_size * s_1_0_matrix.shape[0], 0:map_size] = 0

    for i in range(buffer_size):
        # assert other blocks
        s_1_0_shape = s_1_0_matrix.shape[0]
        result[
            map_size * s_1_0_shape + i * map_size * s_1_0_shape * ph2_size:map_size * s_1_0_shape + (
                        i + 1) * map_size * s_1_0_shape * ph2_size,
            map_size + i * map_size * ph2_size:map_size + (i + 1) * map_size * ph2_size,

            ] = 0

    assert np.all(result == 0)
