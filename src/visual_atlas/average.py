# ================================================================
# 0. Section: Imports
# ================================================================
import nibabel as nib
import numpy as np

from.io_utils import get_nifty_paths_from_folder



# ================================================================
# 1. Section: Average Volume of Nifti Files Brains
# ================================================================
def get_average_volume(brains_folder: str) -> float:
    """Calculate the average volume of brain MRIs from a specified folder.

    This function iterates through all NIfTI files within a given directory,
    calculates the volume for each brain scan, and then computes the average
    of these volumes. The volume of each scan is determined by counting the
    number of non-zero voxels and multiplying by the volume of a single voxel.

    Parameters
    ----------
    brains_folder : str
        The file path to the directory containing the brain MRI scans in
        NIfTI format (.nii or .nii.gz).

    Returns
    -------
    float
        The average volume of the brains in the folder. The units are
        determined by the voxel dimensions in the NIfTI header (e.g., mm³).

    Notes
    -----
    - This function depends on the `nibabel` and `numpy` libraries.
    - It assumes the presence of a helper function `get_nifty_paths_from_folder`
      that returns a list of paths to NIfTI files.
    - A brain mask is created by considering all voxels with a value greater
      than zero as part of the brain tissue.

    Examples
    --------
    >>> # Assuming 'path/to/my/brains' contains several .nii.gz files
    >>> # and get_nifty_paths_from_folder is defined.
    >>> average_vol = get_average_volume('path/to/my/brains')
    >>> print(f"The average brain volume is {average_vol:.2f} mm³.")
    The average brain volume is 1450000.00 mm³.
    """
    
    mri_paths = get_nifty_paths_from_folder(brains_folder)

    volumes = []
    for mri_path in mri_paths:
        # Extract the MRI data
        mri = nib.load(mri_path)
        mri_arr = mri.get_fdata()

        # Get ingredients for volume calculation
        mri_mask = np.where(mri_arr > 0, 1, 0)
        voxel_size = np.round(mri.header.get_zooms(), 3)
        voxel_volume = voxel_size[0] * voxel_size[1] * voxel_size[2]

        # Calculate brain volume
        brain_volume = np.sum(mri_mask) * voxel_volume
        volumes.append(brain_volume)

        # Close the Nifti file
        mri, mri_arr, mri_mask = None, None, None

    # Calculate average volume
    average_volume = np.mean(volumes)

    return average_volume