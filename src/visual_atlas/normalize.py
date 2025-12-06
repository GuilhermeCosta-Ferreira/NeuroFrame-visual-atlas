# ================================================================
# 0. Section: Imports
# ================================================================
import numpy as np



# ================================================================
# 1. Section: Normalize all brains
# ================================================================
def normalize(mri: np.ndarray) -> np.ndarray:
    """Normalize the MRI data to have zero mean and unit variance."""

    mean = np.mean(mri)
    std = np.std(mri)
    normalized_mri = (mri - mean) / std
    return normalized_mri