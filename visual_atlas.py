from src.visual_atlas import *

import nibabel as nib
import matplotlib.pyplot as plt

# First step: initialize dataset
initialize_brains_dataset("input")

# Second step: Global alignment registration
global_alignment_registration("output/split_brains")