# ================================================================
# 0. Section: Imports
# ================================================================
import os

import nibabel as nib
import numpy as np



# ================================================================
# 1. Section: Brain Splitting and Mirroring
# ================================================================
def split_single_brain(mri: np.ndarray, split_axis: int = 2) -> tuple:
    """Splits a 3D MRI brain scan into two separate, symmetrical brain volumes.
    
    This function divides a brain MRI array along a specified axis into left and
    right hemispheres. Each hemisphere is then completed by mirroring its data
    to create a full, symmetrical brain volume. The right hemisphere is flipped
    to match the orientation of the left.

    Parameters
    ----------
    mri : np.ndarray
        The 3D numpy array representing the brain MRI scan.
    split_axis : int, optional
        The axis along which to split the brain, by default 2. This is
        typically the sagittal axis.

    Returns
    -------
    tuple
        A tuple containing two numpy arrays:
        - left_brain (np.ndarray): The reconstructed full brain based on the
          left hemisphere.
        - right_brain (np.ndarray): The reconstructed full brain based on the
          right hemisphere, flipped to match the left's orientation.

    Notes
    -----
    - This function relies on a helper function `fill_missing_half` to
      complete the hemispheres by mirroring.
    - The right hemisphere is flipped along the `split_axis` so that both
      output brains have a consistent orientation (e.g., both are "left-like").
    """
    
    # Get the middle line index along the split axis
    middle_line = mri.shape[split_axis] // 2
    mri_split_indices = np.indices(mri.shape)[split_axis]

    # Split the brain into left and right halves (right half will be flipped)
    left_brain = np.where(mri_split_indices < middle_line, mri, 0)
    right_brain = np.where(mri_split_indices >= middle_line, mri, 0)
    right_brain = np.flip(right_brain, axis=split_axis)

    # Fill in the missing halves by mirroring
    left_brain = fill_missing_half(left_brain, split_axis)
    right_brain = fill_missing_half(right_brain, split_axis)
    
    return left_brain, right_brain


# ──────────────────────────────────────────────────────
# 1.1 Subsection: Fill Missing Half
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
# 1.2 Subsection: Saving System for the Split Brains
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


