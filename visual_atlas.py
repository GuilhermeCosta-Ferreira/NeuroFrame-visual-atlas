from src.visual_atlas import *

import nibabel as nib
import matplotlib.pyplot as plt

#initialize_brains_dataset("input")
vol = get_average_volume("output/split_brains")
print(f"Average brain volume: {vol} mm³")

reference_brain = pick_closest_to_average("output/split_brains", vol)
print(f"Closest brain to average is located at: {reference_brain}")

initial = nib.load("input/t206.nii.gz").get_fdata()
coarse_arr, coarse_tx = coarse_registration(reference_brain, "input/t206.nii.gz")
fixed_arr, rigid_tx = rigid_registration(reference_brain, "input/t206.nii.gz", transform=coarse_tx)
affine_arr, affine_tx = affine_registration(reference_brain, "input/t206.nii.gz", transform=rigid_tx)

print(f"Coarse Transform: {coarse_tx.GetParameters()}")
print(f"Rigid Transform: {rigid_tx.GetParameters()}")
print(f"Affine Transform: {affine_tx.GetParameters()}")

plt.figure(figsize=(12, 6))
plt.subplot(2, 3, 1)
plt.title("Original Brain")
plt.imshow(initial[:, :, initial.shape[2] // 2], cmap='gray')
plt.axis('off')
plt.subplot(2, 3, 2)
plt.title("Before Rigid Registration")
plt.imshow(coarse_arr[:, :, coarse_arr.shape[2] // 2], cmap='gray')
plt.axis('off')
plt.subplot(2, 3, 3)
plt.title("After Rigid Registration")
plt.imshow(fixed_arr[:, :, fixed_arr.shape[2] // 2], cmap='gray')
plt.axis('off')
plt.subplot(2, 3, 4)
plt.title("After Affine Registration")
plt.imshow(affine_arr[:, :, affine_arr.shape[2] // 2], cmap='gray')
plt.axis('off')
plt.tight_layout()
plt.show()