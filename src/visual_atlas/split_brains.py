# ================================================================
# 0. Section: Imports
# ================================================================
import os

import nibabel as nib
import numpy as np
from tqdm import tqdm

from .io_utils import get_nifty_paths_from_folder



# ================================================================
# 1. Section: Split all brains
# ================================================================
def split_all_brains(brains_folder: str, split_axis: int = 2, output_folder: str = 'output/split_brains/', **kwargs) -> None:
    """Processes a folder of NIfTI brain scans, splitting each into two halves.

    This function iterates through all NIfTI files found in the specified `brains_folder`.
    For each file, it loads the MRI data, splits it into two halves (e.g., left and
    right hemispheres) along the given `split_axis`, and saves these halves as new
    NIfTI files in the `output_folder`.

    Parameters
    ----------
    brains_folder : str
        The path to the directory containing the NIfTI brain scans (.nii or .nii.gz)
        to be processed.
    split_axis : int, optional
        The axis along which to split the 3D brain volume. Default is 2, which
        typically corresponds to the sagittal axis for splitting into left and
        right hemispheres.
    output_folder : str, optional
        The path to the directory where the resulting split brain files will be
        saved. Default is 'output/split_brains/'.
    **kwargs : dict
        Optional keyword arguments:
        - verbose : int, optional
            Controls the verbosity of the output.
            - 0: No output.
            - 1: Prints a final completion message.
            - 2: Displays a progress bar during processing.
            Default is 2.

    Returns
    -------
    None
        This function does not return any value. It saves the processed files
        to disk.

    Notes
    -----
    - The function relies on `get_nifty_paths_from_folder` to find the input files,
      `split_single_brain` to perform the splitting logic, and `save_split_brain`
      to handle file saving.
    - Memory is managed by clearing variables within the loop to handle large
      datasets.
    """
    
    verbose = kwargs.get('verbose', 2)

    mri_paths = get_nifty_paths_from_folder(brains_folder)

    for mri_path in tqdm(mri_paths, desc="Processing MRIs", disable=verbose < 2):
        mri = nib.load(mri_path)
        mri_arr = mri.get_fdata()

        # Split the brain and mirror halves
        left_brain, right_brain = split_single_brain(mri_arr, split_axis=split_axis)

        # Save the split brains
        save_split_brain(mri, (left_brain, right_brain), mri_path, output_folder)

        # Close the NIfTI file
        left_brain, right_brain, mri_arr, mri = None, None, None, None
    if verbose >= 1: print(f"✅ All brains processed and saved in: {output_folder}", flush=True)

        
# ──────────────────────────────────────────────────────
# 1.1 Subsection: Individual Brain Splitting
# ──────────────────────────────────────────────────────
def split_single_brain(mri: np.ndarray, split_axis: int = 2) -> tuple:
    """Split the brains and mirror them along the mid-sagittal plane."""

    middle_line = mri.shape[split_axis] // 2
    mri_split_indices = np.indices(mri.shape)[split_axis]

    left_brain = np.where(mri_split_indices < middle_line, mri, 0)
    right_brain = np.where(mri_split_indices >= middle_line, mri, 0)
    right_brain = np.flip(right_brain, axis=split_axis)

    # Fill in the missing halves by mirroring
    left_brain = fill_missing_half(left_brain, split_axis)
    right_brain = fill_missing_half(right_brain, split_axis)
    

    return left_brain, right_brain


# ──────────────────────────────────────────────────────
# 1.2 Subsection: Fill Missing Half
# ──────────────────────────────────────────────────────
def fill_missing_half(brain_half: np.ndarray, split_axis: int = 2) -> np.ndarray:
    """Fill in the missing half of a brain by mirroring the existing half."""
    
    middle_line = brain_half.shape[split_axis] // 2

    if split_axis == 0:
        brain_half[middle_line:, :, :] = np.flip(brain_half[:middle_line, :, :], axis=0)
    elif split_axis == 1:
        brain_half[:, middle_line:, :] = np.flip(brain_half[:, :middle_line, :], axis=1)
    elif split_axis == 2:
        brain_half[:, :, middle_line:] = np.flip(brain_half[:, :, :middle_line], axis=2)

    return brain_half


# ──────────────────────────────────────────────────────
# 1.3 Subsection: Saving System for the Split Brains
# ──────────────────────────────────────────────────────
def save_split_brain(mri: nib.nifti1, brain_halves: tuple, input_path: str, output_folder: str) -> None:
    """Save the split brain halves as separate NIfTI files."""

    # Create NIfTI images for both halves
    left_brain, right_brain = brain_halves
    left_img = nib.Nifti1Image(left_brain, affine=mri.affine)
    right_img = nib.Nifti1Image(right_brain, affine=mri.affine)

    # Generate output file paths
    base_filename = os.path.basename(input_path).replace('.nii.gz', '').replace('.nii', '')
    left_output_path = os.path.join(output_folder, f"{base_filename}_left.nii.gz")
    right_output_path = os.path.join(output_folder, f"{base_filename}_right.nii.gz")

    # Ensure output directory exists
    os.makedirs(output_folder, exist_ok=True)

    nib.save(left_img, left_output_path)
    nib.save(right_img, right_output_path)


