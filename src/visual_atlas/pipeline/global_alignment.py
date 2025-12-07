# ================================================================
# 0. Section: Imports
# ================================================================
import pickle
import os

import nibabel as nib
from tqdm import tqdm
import numpy as np

from ..io_utils import get_nifty_paths_from_folder
from ..average import get_average_volume, pick_closest_to_average
from ..registration import coarse_registration, rigid_registration, affine_registration



# ================================================================
# 1. Section: Initialize the dataset
# ================================================================
def global_alignment_registration(brains_path: str, output_path: str = 'output/general_registration/', **kwargs) -> None:
    """Performs global alignment on a collection of brain MRI scans.
    
    This function orchestrates the global registration pipeline for a set of brain
    MRI scans located in a specified directory. It automatically selects a
    reference brain (the one with a volume closest to the dataset's average)
    and aligns all other brains to it using a sequential, multi-stage
    registration process: coarse (moments-based), rigid, and finally affine.
    The aligned brain volumes are saved as new NIfTI files. Additionally, the
    final affine transformation for each brain, the computed average brain, and
    the chosen reference brain are saved for subsequent use.

    Parameters
    ----------
    brains_path : str
        The file path to the directory containing the input brain MRI scans,
        expected to be in NIfTI format.
    output_path : str, optional
        The directory path where the aligned brain volumes and transformation
        data will be saved. Defaults to 'output/general_registration/'.
    **kwargs : dict, optional
        Optional keyword arguments.
        - verbose : int, optional
            Controls the verbosity of the output.
            - 0: No output.
            - 1: Prints a final confirmation message.
            - 2: Shows a progress bar during registration and prints the
                 final message.
            Default is 2.

    Returns
    -------
    None
        This function does not return any value. It saves the registration
        results to the specified output directory.

    Notes
    -----
    - The function creates the `output_path` directory if it does not exist.
    - The registration pipeline consists of three steps:
        1. Coarse registration based on image moments.
        2. Rigid registration.
        3. Affine registration.
    - The output directory will contain the aligned NIfTI files and a file
      (saved by `save_global_transforms`) containing the average brain, the
      reference brain, and the dictionary of transformations.
      
    Examples
    --------
    >>> # Assuming a directory './data/brains' contains multiple NIfTI files
    >>> global_alignment_registration(
    ...     brains_path='./data/brains',
    ...     output_path='./results/aligned_brains',
    ...     verbose=2
    ... )
    ✅ All brains globally aligned and saved in: ./results/aligned_brains
    """

    verbose = kwargs.get("verbose", 2)

    # Define the reference brain as the one closest to the average volume
    average_volume = get_average_volume(brains_path)
    reference_brain = pick_closest_to_average(brains_path, average_volume)

    # Get all brain MRI paths
    mri_paths = get_nifty_paths_from_folder(brains_path)
    os.makedirs(output_path, exist_ok=True)

    # Iterate through each brain MRI and perform registration
    brain_volumes = []
    global_transforms = {}
    for mri_path in tqdm(mri_paths, desc="Global Alignment Registration", disable=(verbose < 2)):
        # Coarse Moments based registration → Rigid → Affine
        _, coearse_tx = coarse_registration(reference_brain, mri_path)
        _, rigid_tx = rigid_registration(reference_brain, mri_path, transform=coearse_tx)
        affine_arr, affine_tx = affine_registration(reference_brain, mri_path, transform=rigid_tx)

        # Store the final registration results
        global_transforms[mri_path] = affine_tx
        brain_volumes.append(affine_arr)

        # Save the registered brain MRI
        registered_img = nib.Nifti1Image(affine_arr, affine=nib.load(mri_path).affine)
        output_file_path = f"{output_path}{mri_path.split('/')[-1]}"
        nib.save(registered_img, output_file_path)

    average_brain = np.mean(brain_volumes, axis=0)

    save_global_transforms(average_brain, reference_brain, global_transforms, output_path)
    
    if verbose > 0: print(f"✅ All brains globally aligned and saved in: {output_path}", flush=True)


# ──────────────────────────────────────────────────────
# 1.1 Subsection: Saving System for the Global Transforms
# ──────────────────────────────────────────────────────
def save_global_transforms(average_brain: np.ndarray, reference_brain: str, transforms: dict, output_path: str) -> None:
    """Save the global transforms dictionary to a pickle file and the average brain to a nifty."""

    # Ensure output directory exists
    os.makedirs(output_path, exist_ok=True)

    # Save the average brain after registration
    average_img = nib.Nifti1Image(average_brain, affine=nib.load(reference_brain).affine)
    nib.save(average_img, f"{output_path}average_brain.nii.gz")

    # Save the global transforms
    with open(f"{output_path}global_transforms.pkl", "wb") as f: pickle.dump(transforms, f)