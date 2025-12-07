# ================================================================
# 0. Section: Imports
# ================================================================
import numpy as np
import scipy.ndimage as ndi

from .physics import get_axis_rotation_matrix



# ================================================================
# 1. Section: Align all brains to PCA reference
# ================================================================
def reference_align_single_brain(mri: np.ndarray) -> np.ndarray:
    """Aligns a 3D brain MRI to a standard reference orientation.

    This function standardizes the orientation of a brain MRI volume by aligning
    its principal axes, determined by Principal Component Analysis (PCA), with a
    predefined reference coordinate system. The rotation is applied around the
    center of the MRI volume.

    Parameters
    ----------
    mri : np.ndarray
        A 3D NumPy array representing the input brain MRI volume.

    Returns
    -------
    np.ndarray
        A new 3D NumPy array containing the aligned brain MRI. The dimensions
        of the output array are the same as the input array.

    Notes
    -----
    - The function relies on helper functions `get_brain_pca_components`,
      `get_reference_axis`, and `get_axis_rotation_matrix` to compute the
      necessary transformation.
    - The core transformation is performed by `scipy.ndimage.affine_transform`
      using linear interpolation (`order=1`).
    - Areas outside the original volume after rotation are filled with a
      constant value of 0.0.
    """
    
    mri_pca = get_brain_pca_components(mri)
    target_axes = get_reference_axis(mri_pca)

    # Get the rotation matrix for the axis transformation
    R = get_axis_rotation_matrix(mri_pca, target_axes)

    # Build a 3x3 linear part + center of rotation around the image center
    center = 0.5 * (np.array(mri.shape) - 1)
    R_inv = R.T
    offset = center - R_inv @ center
    mri_aligned = ndi.affine_transform(mri, R_inv, offset=offset, order=1, mode="constant", cval=0.0)

    return mri_aligned
    

# ──────────────────────────────────────────────────────
# 1.1 Subsection: PCA Components of the Brain
# ──────────────────────────────────────────────────────
def get_brain_pca_components(mri: np.ndarray) -> np.ndarray:
    """Compute the PCA components of the brain MRI data. 1PC is AP, 2PC is LR, 3PC is IS."""
    mri_mask = np.where(mri > 0, 1, 0)

    coords = np.column_stack(np.where(mri_mask > 0))
    coords_mean = np.mean(coords, axis=0)
    centered_coords = coords - coords_mean

    cov_matrix = np.cov(centered_coords, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

    sorted_indices = np.argsort(eigenvalues)[::-1]
    principal_components = eigenvectors[:, sorted_indices]

    return principal_components


# ──────────────────────────────────────────────────────
# 1.2 Subsection: Reference Axes
# ──────────────────────────────────────────────────────
def get_reference_axis(target_axis: np.ndarray) -> np.ndarray:
    """Get the target reference axes for alignment: AP, LR, IS in matrix form."""
    is_target = target_axis[0, 2]
    ap_target = target_axis[1, 0]
    lr_target = target_axis[2, 1]

    is_value = 1.0 if is_target >= 0 else -1.0
    ap_value = 1.0 if ap_target >= 0 else -1.0
    lr_value = 1.0 if lr_target >= 0 else -1.0

    is_axis = np.array([is_value, 0, 0])
    ap_axis = np.array([0, ap_value, 0])
    lr_axis = np.array([0, 0, lr_value])

    target_axes = np.column_stack((ap_axis, lr_axis, is_axis))
    return target_axes