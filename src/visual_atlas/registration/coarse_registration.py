# ================================================================
# 0. Section: Imports
# ================================================================
import SimpleITK as sitk
import numpy as np



# ================================================================
# 1. Section: Coarse Registration of Two Brain MRIs
# ================================================================
def coarse_registration(reference_brain_path: str, moving_brain_path: str) -> tuple[np.ndarray, sitk.Transform]:
    """Perform a coarse registration of a moving brain MRI to a reference brain MRI."""

    fixed = sitk.ReadImage(reference_brain_path, sitk.sitkFloat32)
    moving = sitk.ReadImage(moving_brain_path, sitk.sitkFloat32)

    # 2) Moments based initial transform (rigid, 3D)
    initial_transform = sitk.CenteredTransformInitializer(
        fixed,
        moving,
        sitk.Euler3DTransform(),  # 3D rotation + translation
        sitk.CenteredTransformInitializerFilter.MOMENTS
    )

    # 3) Resample moving onto fixed grid with this coarse transform
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetTransform(initial_transform)
    resampler.SetDefaultPixelValue(0.0)

    moving_coarse = resampler.Execute(moving)

    moving_coarse = sitk.GetArrayFromImage(moving_coarse)
    return np.transpose(moving_coarse, (2, 1, 0)), initial_transform
    
