import numpy as np

from analytics import q_matrix_calculation
from analytics.solution.distribution import resolve_p_0_and_p_1, resolve_matrix_square_equation, resolve_p_0_and_p_1, \
    calculate_p_i_vectors, calculate_p_i_distribution, calculate_q_j_distribution, find_r_matrix
from analytics.solution.distribution_model import DistributionModel
from analytics.solution.loss_probability import calculate_loss_prob


class ForkJoinModel:

    def __init__(self, d_matrix: tuple[np.ndarray, np.ndarray],
                 ph1: tuple[np.ndarray, np.ndarray],
                 ph2: tuple[np.ndarray, np.ndarray],
                 buffer_size: int
                 ):
        self.d0_matrix = d_matrix[0]
        self.d1_matrix = d_matrix[1]
        self.map_size = d_matrix[0].shape[0]

        assert self.d0_matrix.shape == self.d1_matrix.shape

        self.ph1 = ph1
        self.ph1_size = ph1[0].shape[0]

        self.ph2 = ph2
        self.ph2_size = ph2[0].shape[0]

        self.buffer_size = buffer_size

    def calculate(self):
        self.q0 = self._calculate_q0_matrix()
        self.q1 = self._calculate_q1_matrix()
        self.q2 = self._calculate_q2_matrix()

        self.r = find_r_matrix(
            self.q2,
            self.q1,
            self.q0,
        )

        self.q00 = self._calculate_q00_matrix()
        self.q01 = self._calculate_q01_matrix()
        self.q10 = self._calculate_q10_matrix()

        self.solution = resolve_p_0_and_p_1(
            self.q00,
            self.q10,
            self.q01,
            self.q1,
            self.q0,
            self.r
        )

        # p vactors
        self.p_vectors = list(self.solution) + calculate_p_i_vectors(self.solution[1], self.r)

        # distributions
        self.p_distribution = DistributionModel(
            calculate_p_i_distribution(self.p_vectors)
        )
        self.q_distribution = DistributionModel(
            calculate_q_j_distribution(
                self.p_vectors,
                self.buffer_size,
                self.map_size,
                self.ph1_size,
                self.ph2_size
            )
        )

        self.loss_prob = calculate_loss_prob(
            (self.d0_matrix, self.d1_matrix),
            self.p_vectors,
            self.buffer_size,
            self.map_size,
            self.ph1_size,
            self.ph2_size
        )

    def get_metrics(self):
        return {
            'loss_prob': self.loss_prob,
            'q_mean': self.q_distribution.mean(),
            'p_mean': self.p_distribution.mean(),
        }


    def _calculate_q0_matrix(self):
        return q_matrix_calculation.calculate_q_0_matrix(
            self.buffer_size,
            self.ph2_size,
            self.map_size,
            self.ph1
        )

    def _calculate_q1_matrix(self):
        return q_matrix_calculation.calculate_q_1_matrix(
            self.buffer_size,
            (self.d0_matrix, self.d1_matrix),
            self.ph1,
            self.ph2,
        )

    def _calculate_q2_matrix(self):
        return q_matrix_calculation.calculate_q_2_matrix(
            self.buffer_size,
            self.ph1_size,
            self.d1_matrix,
            self.ph2[1],
        )

    def _calculate_q00_matrix(self):
        return q_matrix_calculation.calculate_q_0_0_matrix(
            self.buffer_size,
            (self.d0_matrix, self.d1_matrix),
            self.ph2,
        )

    def _calculate_q10_matrix(self):
        return q_matrix_calculation.calculate_q_1_0_matrix(
            self.buffer_size,
            self.ph2_size,
            self.map_size,
            self.ph1[0],
        )

    def _calculate_q01_matrix(self):
        return q_matrix_calculation.calculate_q_0_1_matrix(
            self.buffer_size,
            self.d1_matrix,
            (self.ph1[1], self.ph2[1])
        )

