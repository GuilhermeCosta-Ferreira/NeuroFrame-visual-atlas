# ================================================================
# 0. Section: Imports
# ================================================================
import os

import nibabel as nib
from tqdm import tqdm

from ..io_utils import get_nifty_paths_from_folder
from ..normalize import normalize
from ..split_brains import split_single_brain
from ..reference_align import reference_align_single_brain



# ================================================================
# 1. Section: Initialize the dataset
# ================================================================
def initialize_brains_dataset(brains_folder: str, split_axis: int = 2, output_folder: str = 'output/split_brains/', **kwargs) -> None:
    """Processes a dataset of brain MRIs by normalizing, aligning, and splitting them.

    This function iterates through all NIfTI files in a specified input folder.
    For each MRI, it performs a series of preprocessing steps:
    1. Normalizes the intensity values of the MRI data.
    2. Aligns the brain to a standard reference orientation.
    3. Splits the brain into two hemispheres (left and right) along a specified axis.
    The resulting processed hemispheres are then saved as new NIfTI files in the
    specified output directory.

    Parameters
    ----------
    brains_folder : str
        The path to the directory containing the input NIfTI brain scans
        (e.g., '.nii' or '.nii.gz' files).
    split_axis : int, optional
        The axis along which the brain will be split into two hemispheres.
        Default is 2.
    output_folder : str, optional
        The path to the directory where the processed and split brain hemispheres
        will be saved. Default is 'output/split_brains/'.
    **kwargs : dict
        Optional keyword arguments.
        - verbose : int, optional
            Controls the verbosity of the output.
            - 0: No output.
            - 1: Prints a final confirmation message.
            - 2: Displays a progress bar during processing and prints the final
                 message.
            Default is 2.

    Returns
    -------
    None
        This function does not return any value. It saves the processed files
        to disk.

    Notes
    -----
    - The function relies on external helper functions: `get_nifty_paths_from_folder`,
      `normalize`, `reference_align_single_brain`, `split_single_brain`, and
      `save_split_brain`.
    - It processes each MRI file found in `brains_folder` and creates
      corresponding output files for the left and right hemispheres in
      `output_folder`.
    - Memory is managed by clearing variables after processing each brain to handle
      large datasets.
    """
    
    # Extract the optional parameters
    verbose = kwargs.get('verbose', 2)

    # Get all NIfTI file paths from the input folder
    mri_paths = get_nifty_paths_from_folder(brains_folder)

    # Process each MRI file
    for mri_path in tqdm(mri_paths, desc="Processing MRIs", disable=verbose < 2):
        mri = nib.load(mri_path)
        mri_arr = mri.get_fdata()

        # Normalize the MRI data (0 - 255)
        mri_arr = normalize(mri_arr)

        # Center and align the brain (PCA-based)
        mri_arr = reference_align_single_brain(mri_arr)

        # Split the brain and mirror halves
        left_brain, right_brain = split_single_brain(mri_arr, split_axis=split_axis)

        # Save the split brains and close the NIfTI file
        save_split_brain(mri, (left_brain, right_brain), mri_path, output_folder)
        left_brain, right_brain, mri_arr, mri = None, None, None, None
    if verbose >= 1: print(f"✅ All brains processed and saved in: {output_folder}", flush=True)


# ──────────────────────────────────────────────────────
# 1.1 Subsection: Saving System for the Initialization
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


