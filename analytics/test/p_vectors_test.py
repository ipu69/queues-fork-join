import time

import numpy as np

import analytics.test_suit as test_suit
from analytics.ergodicity.ergodicity import get_map_lambda, get_ph_mu, get_y_j, verify_ergodicity
from analytics.q_matrix_calculation import calc_s_0_matrix

from analytics.solution.distribution import resolve_matrix_square_equation, resolve_p_0_and_p_1, calculate_p_i_vectors, \
    calculate_p_i_distribution, find_r_matrix, calculate_q_j_distribution
from analytics import q_matrix_calculation

from analytics.solution.distribution_model import DistributionModel
from analytics.matrix_helper import concat_diag_blocks, concat_above_diag_blocks, concat_sub_diag_blocks

from analytics.math_helper import kron, kron_sum
from analytics.solution.loss_probability import calculate_loss_prob


def test():
    cur_time = time.time()
    d_matrices = test_suit.d_matrices
    ph = test_suit.ph
    buffer_size = test_suit.buffer_size
    map_size = d_matrices[0].shape[0]
    ph1_size = ph[0].shape[0]
    ph2_size = ph[0].shape[0]

    q_0 = q_matrix_calculation.calculate_q_0_matrix(
        buffer_size,
        ph[0].shape[0],
        d_matrices[0].shape[0],
        ph,
    )

    q_1 = q_matrix_calculation.calculate_q_1_matrix(
        buffer_size,
        d_matrices,
        ph,
        ph
    )

    q_2 = q_matrix_calculation.calculate_q_2_matrix(
        buffer_size,
        ph[0].shape[0],
        d_matrices[1],
        ph[1],
    )

    r = resolve_matrix_square_equation(q_0, q_1, q_2)

    q_0_0 = q_matrix_calculation.calculate_q_0_0_matrix(
        buffer_size,
        d_matrices,
        ph
    )

    q_0_1 = q_matrix_calculation.calculate_q_0_1_matrix(
        buffer_size,
        d_matrices[1],
        (
            ph[1],
            ph[1],
        )
    )

    q_1_0 = q_matrix_calculation.calculate_q_1_0_matrix(
        buffer_size,
        ph[0].shape[0],
        d_matrices[0].shape[0],
        ph[0]
    )

    assert np.allclose(np.sum(
        np.hstack((
            q_0_0, q_0_1
        )),
        axis=1
    ), 0)

    assert np.allclose(np.sum(
        np.hstack((
            q_1_0, q_1, q_2
        )),
        axis=1
    ), 0)

    assert np.allclose(np.sum(
        np.hstack((
            q_0,
            q_1,
            q_2
        )),
        axis=1
    ), 0)

    full_generator_part: np.ndarray = concat_diag_blocks([q_0_0, q_1, q_1], additional_right_zeros=q_2.shape[1]) +\
                          concat_above_diag_blocks([q_0_1, q_2, q_2], first_zero_left_block_width=q_0_0.shape[1]) +\
                          concat_sub_diag_blocks([q_1_0, q_0], first_zero_above_block_height=q_0_0.shape[0], last_zero_right_block_width=q_1.shape[1]+q_2.shape[1])

    assert np.all(full_generator_part.diagonal() < 0)
    np.fill_diagonal(full_generator_part, 1)
    assert np.all(full_generator_part >= 0)

    solution = resolve_p_0_and_p_1(
        q_0_0,
        q_1_0,
        q_0_1,
        q_1,
        q_0,
        r
    )

    assert min(solution[0]) >= 0
    assert max(solution[0]) <= 1
    assert sum(solution[0]) <= 1


    assert min(solution[1]) >= 0
    assert max(solution[1]) <= 1
    assert sum(solution[1]) <= 1

    assert sum(np.concatenate((solution[0], solution[1]))) <= 1

    p_vectors = list(solution) + calculate_p_i_vectors(solution[1], r)

    p_distribution = DistributionModel(
        calculate_p_i_distribution(p_vectors)
    )

    q_distribution = DistributionModel(
        calculate_q_j_distribution(
            p_vectors,
            buffer_size,
            map_size,
            ph1_size,
            ph2_size
        )
    )

    loss_prob = calculate_loss_prob(
        d_matrices,
        p_vectors,
        buffer_size,
        map_size,
        ph1_size,
        ph2_size
    )

    print(q_distribution.distribution)
    print(q_distribution.mean())
    print(loss_prob)
    print(solution)

    print(np.dot(solution[0], q_0_0) + np.dot(solution[1], q_1_0))
    print(np.dot(solution[0], q_0_1) + np.dot(solution[1], q_1 + np.dot(r, q_0)))
    print(1 - (np.sum(p_vectors[0]) + np.sum(np.dot(p_vectors[1], np.linalg.inv(np.eye(r.shape[0]) - r)))))

    print(f'Execution takes {time.time() - cur_time}')
