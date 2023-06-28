import numpy as np
from analytics.math_helper import kron, kron_sum
from analytics.matrix_helper import concat_sub_diag_blocks, \
    concat_diag_blocks, \
    concat_above_diag_blocks


def calculate_q_0_0_matrix(buffer_size: int,
                           d_matrices: tuple[np.ndarray, np.ndarray],
                           ph2: tuple[np.ndarray, np.ndarray]) -> np.ndarray:

    assert buffer_size > 0

    map_size: int = d_matrices[0].shape[0]

    d0_matrix = d_matrices[0]
    d1_matrix = d_matrices[1]

    s2_matrix = ph2[0]
    beta2_vector = ph2[1]

    _kron_matrix = kron_sum(d0_matrix, s2_matrix)
    _s_0_2 = calc_s_0_matrix(s2_matrix)

    last_diag_block = kron_sum(d0_matrix + d1_matrix, s2_matrix)

    main_sub_diag_block = kron(np.eye(map_size), np.dot(_s_0_2, beta2_vector))

    return concat_diag_blocks(
        [d0_matrix] + [_kron_matrix for i in range(buffer_size - 1)] + [last_diag_block]
    ) + concat_sub_diag_blocks(
            [kron(np.eye(map_size), _s_0_2)] + [main_sub_diag_block for i in range(buffer_size - 1)],
            first_zero_above_block_height=d0_matrix.shape[0],
            last_zero_right_block_width=last_diag_block.shape[1]
    )


# verified (there are test for this function)
def calculate_q_0_1_matrix(buffer_size: int,
                           d1_matrix: np.ndarray,
                           beta_vectors: tuple[np.ndarray, np.ndarray],
                           ) -> np.ndarray:
    assert buffer_size > 0

    w = d1_matrix.shape[0]

    beta_1_vector = beta_vectors[0]
    beta_2_vector = beta_vectors[1]

    ph2_size = beta_2_vector.shape[1]
    ph1_size = beta_1_vector.shape[1]

    first_block = kron(d1_matrix, beta_1_vector, beta_2_vector)
    main_block = kron(d1_matrix, beta_1_vector, np.eye(ph2_size))

    return concat_above_diag_blocks(
        [first_block] + [main_block for i in range(buffer_size - 1)],
        first_zero_left_block_width=w*ph1_size,
        last_zero_block_height=w*ph2_size
    )


def calculate_q_1_0_matrix(buffer_size: int,
                           ph2_size: int,
                           map_size: int,
                           s1_matrix: np.ndarray
                           ) -> np.ndarray:
    _s_0_1 = calc_s_0_matrix(s1_matrix)

    first_block = kron(np.eye(map_size), _s_0_1)
    main_block = kron(first_block, np.eye(ph2_size))

    return concat_diag_blocks(
        [first_block] + [main_block for i in range(buffer_size)]
    )


def calculate_q_0_matrix(buffer_size: int,
                         ph2_size: int,
                         map_size: int,
                         ph1: tuple[np.ndarray, np.ndarray]
                         ) -> np.ndarray:
    s1_matrix: np.ndarray = ph1[0]
    beta_1_vector: np.ndarray = ph1[1]

    _s_0_1 = calc_s_0_matrix(s1_matrix)

    multiplication_result = np.dot(_s_0_1, beta_1_vector)

    first_block = kron(np.eye(map_size), multiplication_result)
    main_block = kron(first_block, np.eye(ph2_size))

    return concat_diag_blocks(
        [first_block] + [main_block for i in range(buffer_size)]
    )


def calculate_q_1_matrix(buffer_size: int,
                         d_matrices: tuple[np.ndarray, np.ndarray],
                         ph1: tuple[np.ndarray, np.ndarray],
                         ph2: tuple[np.ndarray, np.ndarray]) -> np.ndarray:

    assert buffer_size > 0

    d0_matrix = d_matrices[0]
    d1_matrix = d_matrices[1]

    map_size: int = d0_matrix.shape[0]
    ph1_size: int = ph1[0].shape[0]

    s1_matrix = ph1[0]

    s2_matrix = ph2[0]
    beta2_vector = ph2[1]

    ph2_size = s2_matrix.shape[0]

    _s_0_1 = calc_s_0_matrix(s1_matrix)
    _s_0_2 = calc_s_0_matrix(s2_matrix)

    main_diag_block = kron_sum(d0_matrix, s1_matrix, s2_matrix)
    main_sub_dig_block = kron(np.eye(map_size*ph1_size), np.dot(_s_0_2, beta2_vector))

    return concat_diag_blocks(
            [kron_sum(d0_matrix, s1_matrix)] + [main_diag_block for i in range(buffer_size - 1)] + [kron_sum(d0_matrix + d1_matrix, s1_matrix, s2_matrix)]
    ) + concat_sub_diag_blocks(
            [kron(np.eye(map_size*ph1_size), _s_0_2)] + [main_sub_dig_block for i in range(buffer_size - 1)],
            first_zero_above_block_height=map_size * ph1_size,
            last_zero_right_block_width=map_size * ph1_size * ph2_size
    )


def calculate_q_2_matrix(buffer_size: int,
                         ph1_size: int,
                         d1_matrix: np.ndarray,
                         beta_2_vector: np.ndarray,
                         ) -> np.ndarray:
    w = d1_matrix.shape[0]

    assert buffer_size > 0

    ph2_size = beta_2_vector.shape[1]
    main_block = kron(d1_matrix, np.eye(ph1_size * ph2_size))
    first_block = kron(d1_matrix, np.eye(ph1_size), beta_2_vector)

    return concat_above_diag_blocks(
        [first_block] + [main_block for i in range(buffer_size - 1)],
        first_zero_left_block_width=ph1_size * w,
        last_zero_block_height=ph1_size * w * ph2_size
    )


def calc_s_0_matrix(s_matrix: np.ndarray) -> np.ndarray:
    return -np.sum(s_matrix, axis=1).reshape(s_matrix.shape[0], 1)
