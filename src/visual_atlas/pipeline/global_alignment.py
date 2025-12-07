# ================================================================
# 0. Section: Imports
# ================================================================
import pickle

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
    """Perform global alignment registration on a dataset of brain MRIs."""

    verbose = kwargs.get("verbose", 0)

    # Define the reference brain as the one closest to the average volume
    average_volume = get_average_volume(brains_path)
    reference_brain = pick_closest_to_average(brains_path, average_volume)

    # Get all brain MRI paths
    mri_paths = get_nifty_paths_from_folder(brains_path)

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
    
    print(f"✅ All brains globally aligned and saved in: {output_path}", flush=True)


# ──────────────────────────────────────────────────────
# 1.1 Subsection: Saving System for the Global Transforms
# ──────────────────────────────────────────────────────
def save_global_transforms(average_brain: np.ndarray, reference_brain: str, transforms: dict, output_path: str) -> None:
    """Save the global transforms dictionary to a pickle file and the average brain to a nifty."""

    # Save the average brain after registration
    average_img = nib.Nifti1Image(average_brain, affine=nib.load(reference_brain).affine)
    nib.save(average_img, f"{output_path}average_brain.nii.gz")
    print(f"✅ Average brain saved in: {output_path}average_brain.nii.gz", flush=True)

    # Save the global transforms
    with open(f"{output_path}global_transforms.pkl", "wb") as f: pickle.dump(transforms, f)
    print(f"✅ Global transforms saved in: {output_path}global_transforms.pkl", flush=True)