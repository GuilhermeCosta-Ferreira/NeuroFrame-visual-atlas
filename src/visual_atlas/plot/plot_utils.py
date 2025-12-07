# ================================================================
# 0. Section: Imports
# ================================================================
import numpy as np



# ================================================================
# 1. Section: Main Slices Extraction
# ================================================================
def get_main_slices(volume: np.ndarray) -> dict:
    """Extracts the main axial, coronal, and sagittal slices from a 3D volume.

    This function takes a 3D NumPy array and returns a dictionary containing the
    three central orthogonal slices. The slices are taken from the middle of each
    respective axis.

    Parameters
    ----------
    volume : np.ndarray
        The 3D input volume, typically a medical image scan, from which to
        extract the central slices.

    Returns
    -------
    dict
        A dictionary with keys 'axial', 'coronal', and 'sagittal'. Each key
        maps to a 2D NumPy array representing the central slice for that
        orientation.

    Examples
    --------
    >>> import numpy as np
    >>> volume_data = np.arange(27).reshape((3, 3, 3))
    >>> slices = get_main_slices(volume_data)
    >>> slices['axial'].shape
    (3, 3)
    >>> slices['coronal'].shape
    (3, 3)
    >>> slices['sagittal'].shape
    (3, 3)
    >>> print(slices['axial'])
    [[ 9 10 11]
     [12 13 14]
     [15 16 17]]
    """
    mid_slices = {
        'axial': volume[volume.shape[0] // 2, :, :],
        'coronal': volume[:, volume.shape[1] // 2, :],
        'sagittal': volume[:, :, volume.shape[2] // 2]
    }
    return mid_slices