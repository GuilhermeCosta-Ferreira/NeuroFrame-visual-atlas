# ================================================================
# 0. Section: Imports
# ================================================================
import SimpleITK as sitk
import numpy as np



# ================================================================
# 1. Section: Composite Registration of Two Brain MRIs
# ================================================================
def composite_registration(reference_brain_path: str, moving_brain: str | np.ndarray, tarnsformations: np.ndarray):
    """Create a composite transform combining multiple registrations."""
    
    fixed  = sitk.ReadImage(reference_brain_path, sitk.sitkFloat32)

    if(isinstance(moving_brain, np.ndarray)): 
        moving = sitk.GetImageFromArray(moving_brain)
        moving.SetSpacing(fixed.GetSpacing())
    else: moving = sitk.ReadImage(moving_brain, sitk.sitkFloat32)


    composite = sitk.CompositeTransform(3)
    for tx in tarnsformations: composite.AddTransform(tx)

    final_resampled = sitk.Resample(
        moving,
        fixed,
        composite,
        sitk.sitkLinear,
        0.0,
        sitk.sitkFloat32,
    )

    composite_array = sitk.GetArrayFromImage(final_resampled)
    composite_array = np.transpose(composite_array, (2, 1, 0))

    return composite_array, composite

