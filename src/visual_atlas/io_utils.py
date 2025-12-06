# ================================================================
# 0. Section: Imports
# ================================================================
import os



# ================================================================
# 1. Section: Nifty Handlers
# ================================================================
def get_nifty_paths_from_folder(folder_path: str) -> list:
    """Get all .nii and .nii.gz file paths from the specified folder."""

    nifty_paths = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.nii') or file.endswith('.nii.gz'):
                nifty_paths.append(os.path.join(root, file))
    return nifty_paths