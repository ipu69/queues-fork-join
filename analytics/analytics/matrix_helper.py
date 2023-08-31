import numpy as np
from typing import List, Tuple


def concat_diag_blocks(blocks,
                       additional_sub_zeros=0,
                       additional_right_zeros=0):
    """
    Function Concatenates blocks into block-diagonal matrix, concatenate matrix by diagonal

    blocks = [np.array([[1,2], [3,4]]), np.array([[5,6], [7,8]])]
    others params by default:

    return result.txt: np.array([[1,2,0,0],
                             [3,4,0,0],
                             [0,0,5,6],
                             [0,0,7,8]])

    Parameters
    ----------
    blocks
    additional_sub_zeros
    additional_right_zeros

    Returns
    -------

    """

    answer = np.array([])

    for index, diag_block in enumerate(blocks):
        if answer.size != 0:
            added_zeros_bottom = np.zeros(
                shape=(diag_block.shape[0], answer.shape[1])
            )
            added_zeros_right = np.zeros(
                shape=(answer.shape[0], diag_block.shape[1])
            )
            answer = np.hstack((
                np.vstack((answer, added_zeros_bottom)),
                np.vstack((added_zeros_right, diag_block))
            ))
        else:
            answer = diag_block

    if additional_sub_zeros > 0:
        answer = add_zero_rows_down(answer, additional_sub_zeros)

    if additional_right_zeros > 0:
        answer = add_zeros_cols_right(answer, additional_right_zeros)

    return answer


def concat_sub_diag_blocks(blocks: List[np.ndarray],
                           first_zero_above_block_height=1,
                           last_zero_right_block_width=0) -> np.ndarray:
    """
    Function concatenates blocks into block-diagonal matrix, concatenate matrix by sub diagonal

    blocks = [np.array([[1,2], [3,4]]), np.array([[5,6], [7,8]])],
    others params by default:

    result.txt = np.array([[0,0,0,0],
                       [1,2,0,0],
                       [3,4,0,0],
                       [0,0,5,6],
                       [0,0,7,8]])

    Parameters
    ----------
    blocks
    first_zero_above_block_height
    last_zero_right_block_width

    Returns
    -------
    np.ndarray
    """

    answer = np.array([])

    for index, block in enumerate(blocks):
        if index == 0:
            zero_sub_block_width = block.shape[1]
            above_zero_block = np.zeros(shape=(first_zero_above_block_height, zero_sub_block_width))
            answer = np.vstack((above_zero_block, block))
        else:
            answer = np.hstack((
                add_zero_rows_down(answer, block.shape[0]),
                add_zero_rows_up(block, answer.shape[0])
            ))

    if last_zero_right_block_width > 0:
        answer = add_zeros_cols_right(answer, last_zero_right_block_width)

    return answer


def concat_above_diag_blocks(blocks,
                             first_zero_left_block_width=1,
                             last_zero_block_height=0):
    """
    Function concatenates blocks into block-diagonal matrix, concatenate blocks by above diagonal

    blocks = [np.array([[1,2], [3,4]]), np.array([[5,6], [7,8]])],
    others params by default:

    return result.txt: np.array([[0,1,2,0,0],
                            [0,3,4,0,0],
                            [0,0,0,5,6],
                            [0,0,0,7,8]])

    Parameters
    ----------
    blocks
    first_zero_left_block_width
    last_zero_block_height

    Returns
    -------
    np.ndarray
    """

    answer = np.array([])

    for index, block in enumerate(blocks):
        if index == 0:
            answer = add_zeros_cols_left(block, first_zero_left_block_width)
        else:
            answer = np.hstack((
                add_zero_rows_down(answer, block.shape[0]),
                add_zero_rows_up(block, answer.shape[0])
            ))

    if last_zero_block_height > 0:
        answer = add_zero_rows_down(answer, last_zero_block_height)

    return answer


def add_zeros_cols_right(matrix: np.ndarray, count_cols: int) -> np.ndarray:
    """
    Function add zero cols from right side (count_cols param) to matrix

    matrix = |1 2 3|
             |4 5 6|
             |7 8 9|

    count_cols = 2

    result.txt = |1 2 3 0 0|
             |4 5 6 0 0|
             |7 8 9 0 0|

    Parameters
    ----------
    matrix
    count_cols

    Returns
    -------
    np.ndarray
    """

    return np.hstack((
        matrix,
        np.zeros(shape=(matrix.shape[0], count_cols))
    ))


def add_zeros_cols_left(matrix, count_cols):
    """
    Function add zero cols from left side (count_cols param) to matrix

    matrix = |1 2 3|
             |4 5 6|
             |7 8 9|

    count_cols = 2

    result.txt = |0 0 1 2 3|
             |0 0 4 5 6|
             |0 0 7 8 9|


    Parameters
    ----------
    matrix
    count_cols

    Returns
    -------
    np.ndarray
    """

    return np.hstack((
        np.zeros(shape=(matrix.shape[0], count_cols)),
        matrix
    ))


def add_zero_rows_down(matrix, rows_count):
    """
    Function add zero cols from down side (count_cols param) to matrix

    matrix = |1 2 3|
             |4 5 6|
             |7 8 9|

    count_cols = 2

    result.txt = |1 2 3|
             |4 5 6|
             |7 8 9|
             |0 0 0|
             |0 0 0|

    Parameters
    ----------
    matrix
    rows_count

    Returns
    -------

    """

    return np.vstack((
        matrix,
        np.zeros(shape=(rows_count, matrix.shape[1]))
    ))


def add_zero_rows_up(matrix, rows_count):
    """
    Function add zero cols from up side (count_cols param) to matrix

    matrix = |1 2 3|
             |4 5 6|
             |7 8 9|

    count_cols = 2

    result.txt =  |0 0 0|
              |0 0 0|
              |1 2 3]
              |4 5 6|
              |7 8 9|

    Parameters
    ----------
    matrix
    rows_count

    Returns
    -------

    """

    return np.vstack((
        np.zeros(shape=(rows_count, matrix.shape[1])),
        matrix
    ))


def align_matrices_shapes(first: np.ndarray, second: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Function align matrices shapes adding zeros from
    right and down sides of matrix

    first = |1 2 3|
            |4 5 6|

    second = |7  8|
             |9 10|
             |0  1|

    return two matrices:

    |1 2 3|       |7 8  0|
    |4 5 6|       |9 10 0|
    |0 0 0|       |0 1  0|

    Parameters
    ----------
    first
    second

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
    """
    if first.shape[0] < second.shape[0]:
        first = add_zero_rows_down(first, second.shape[0] - first.shape[0])
    elif first.shape[0] > second.shape[0]:
        second = add_zero_rows_down(second, first.shape[0] - second.shape[0])

    if second.shape[1] > first.shape[1]:
        first = add_zeros_cols_right(first, second.shape[1] - first.shape[1])
    elif second.shape[1] < first.shape[1]:
        second = add_zeros_cols_right(second, first.shape[1] - second.shape[1])

    return first, second
