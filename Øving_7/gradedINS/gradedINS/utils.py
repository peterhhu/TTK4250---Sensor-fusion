import numpy as np
from mytypes import ArrayLike


def cross_product_matrix(n: ArrayLike, debug: bool = True) -> np.ndarray:
    assert len(n) == 3, f"utils.cross_product_matrix: Vector not of length 3: {n}"
    vector = np.array(n, dtype=float).reshape(3)

    S = np.zeros((3, 3))

    S[0][1] = -vector[2]
    S[1][0] = vector[2]
    S[0][2] = vector[1]
    S[2][0] = -vector[1]
    S[1][2] = -vector[0]
    S[2][1] = vector[0]

    if debug:
        assert S.shape == (
            3,
            3,
        ), f"utils.cross_product_matrix: Result is not a 3x3 matrix: {S}, \n{S.shape}"
        assert np.allclose(
            S.T, -S
        ), f"utils.cross_product_matrix: Result is not skew-symmetric: {S}"

    return S


def taylor_approximate_A(A: np.ndarray, Ts: float, degree: int) -> np.ndarray:
    """Perform taylor approximation of Q.

    Args:
        A (np.ndarray): System matrix
        degree (int): Order of the approximation

    Returns:
        np.ndarray: Discretized A-matrix
    """
    Ad = np.eye(A.shape[0])

    matrix_product = Ad

    i = 1

    while i <= degree:
        matrix_product = matrix_product @ A
        Ad += matrix_product * (Ts ** i) / np.math.factorial(i)
        i += 1

    return Ad

def taylor_approximate_Q(A: np.ndarray, G: np.ndarray, D: np.ndarray, Ts: float, degree: int) -> np.ndarray:
    """Perform taylor approximation of Q.

    Args:
        A (np.ndarray): System matrix
        G (np.ndarray): Noise matrix
        D (np.ndarray): Noise covariance
        degree (int): Order of the approximation

    Returns:
        np.ndarray: Discretized Q-matrix
    """

    exp_A_approx = taylor_approximate_A(A, Ts, degree)
    exp_A_transp_approx = taylor_approximate_A(A.T, Ts, degree)

    Qd = exp_A_approx @ G @ D @ G.T @ exp_A_transp_approx * Ts

    return Qd