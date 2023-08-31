import numpy as np


class ForkJoinModel:

    def __init__(self, d_matrix: tuple[np.ndarray, np.ndarray],
                 ph1: tuple[np.ndarray, np.ndarray],
                 ph2: tuple[np.ndarray, np.ndarray],
                 buffer_size: int
                 ):
        self.d0_matrix = d_matrix[0]
        self.d1_matrix = d_matrix[1]

        self.s1_matrix = ph1[0]
        self.beta1_vector = ph1[1]

        self.s2_matrix = ph2[0]
        self.beta2_vector = ph2[1]

        self.buffer_size = buffer_size

    def calculate(self):
        pass

    def _calculate_generator(self):
        pass

    def _calculate_q0_matrix(self):
        pass

    def _calculate_q1_matrix(self):
        pass

    def _calculate_q2_matrix(self):
        pass

    def _calculate_q00_matrix(self):
        pass

    def _calculate_q10_matrix(self):
        pass

    def _calculate_q01_matrix(self):
        pass

