# ================================================================
# 0. Section: Imports
# ================================================================
import numpy as np



# ================================================================
# 1. Section: Normalize all brains
# ================================================================
def normalize(mri: np.ndarray) -> np.ndarray:
    """Normalize an MRI scan represented as a NumPy array to a 0-255 scale.

    This function performs min-max normalization on an input NumPy array. It
    scales the intensity values to a range of 0 to 255 and converts the data
    type to an unsigned 8-bit integer (`uint8`), which is suitable for image
    visualization.

    Parameters
    ----------
    mri : np.ndarray
        The input MRI scan as a NumPy array.

    Returns
    -------
    np.ndarray
        The normalized MRI scan, with values scaled to the range [0, 255] and
        data type `uint8`.

    Examples
    --------
    >>> import numpy as np
    >>> mri_scan = np.array([[0, 100], [200, 400]], dtype=np.float32)
    >>> normalized_scan = normalize(mri_scan)
    >>> print(normalized_scan)
    [[  0  63]
     [127 255]]
    >>> print(normalized_scan.dtype)
    uint8
    """

    mri_min = np.min(mri)
    mri_max = np.max(mri)
    normalized_mri = (mri - mri_min) / (mri_max - mri_min) * 255
    return normalized_mri.astype(np.uint8)