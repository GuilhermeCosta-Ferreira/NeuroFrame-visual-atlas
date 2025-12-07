# ================================================================
# 0. Section: Imports
# ================================================================
import numpy as np



# ================================================================
# 1. Section: Rotation Matrices
# ================================================================
def get_axis_rotation_matrix(from_axes: np.ndarray, to_axes: np.ndarray) -> np.ndarray:
    """Calculate the rotation matrix to transform one set of axes to another.

    This function computes the rotation matrix `R` that aligns an initial
    coordinate system, defined by `from_axes`, with a target coordinate
    system, defined by `to_axes`. It assumes that the input matrices represent
    orthonormal bases.

    Parameters
    ----------
    from_axes : np.ndarray
        A 3x3 matrix representing the initial coordinate system. Each column
        should be an orthonormal basis vector.
    to_axes : np.ndarray
        A 3x3 matrix representing the target coordinate system. Each column
        should be an orthonormal basis vector.

    Returns
    -------
    np.ndarray
        The 3x3 rotation matrix `R` that transforms vectors from the
        `from_axes` coordinate system to the `to_axes` coordinate system,
        such that `R @ from_axes = to_axes`.

    Notes
    -----
    The function solves the matrix equation `R @ F = T` for `R`, where `F` is
    `from_axes` and `T` is `to_axes`. Assuming `F` is an orthogonal matrix
    (which is true for a valid set of basis vectors), its inverse is its
    transpose (`F_inv = F.T`). Therefore, the rotation matrix `R` is
    calculated as `R = T @ F.T`.

    Examples
    --------
    >>> import numpy as np
    >>> # Define the standard axes (identity matrix)
    >>> standard_axes = np.identity(3)
    >>> # Define axes rotated 90 degrees around the z-axis
    >>> rotated_axes = np.array([
    ...     [0, -1, 0],
    ...     [1,  0, 0],
    ...     [0,  0, 1]
    ... ])
    >>> # Get the rotation matrix to go from standard to rotated axes
    >>> R = get_axis_rotation_matrix(standard_axes, rotated_axes)
    >>> print(np.allclose(R, rotated_axes))
    True
    >>> # Verify that applying the rotation to the original axes yields the target
    >>> print(np.allclose(R @ standard_axes, rotated_axes))
    True
    """
    
    return to_axes @ from_axes.T