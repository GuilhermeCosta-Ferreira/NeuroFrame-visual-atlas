# ================================================================
# 0. Section: Imports
# ================================================================
import SimpleITK as sitk
import numpy as np



# ================================================================
# 1. Section: Affine Registration of Two Brain MRIs
# ================================================================
def affine_registration(reference_brain_path: str, moving_brain: str | np.ndarray, transform: sitk.Transform = None) -> tuple[np.ndarray, sitk.Transform]:
    """Perform an affine registration of a moving brain MRI to a reference brain MRI."""

    fixed  = sitk.ReadImage(reference_brain_path, sitk.sitkFloat32)

    if(isinstance(moving_brain, np.ndarray)): moving = sitk.GetImageFromArray(moving_brain)
    else: moving = sitk.ReadImage(moving_brain, sitk.sitkFloat32)

    moving.SetSpacing(fixed.GetSpacing())

    if(transform is not None): initial_affine = transform
    else:
        initial_affine = sitk.CenteredTransformInitializer(
            fixed,
            moving,
            sitk.AffineTransform(fixed.GetDimension()),
            sitk.CenteredTransformInitializerFilter.MOMENTS
        )

    # 5. Registration setup
    reg = sitk.ImageRegistrationMethod()

    # Metric: MI works well cross-modality
    reg.SetMetricAsMattesMutualInformation(50)
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(0.1)
    reg.SetInterpolator(sitk.sitkLinear)

    # Multi-resolution
    reg.SetShrinkFactorsPerLevel([4, 2, 1])
    reg.SetSmoothingSigmasPerLevel([2, 1, 0])
    reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    # Optimizer
    reg.SetOptimizerAsRegularStepGradientDescent(
        learningRate=1.0,
        minStep=1e-4,
        numberOfIterations=100,
        relaxationFactor=0.5,
    )
    reg.SetOptimizerScalesFromPhysicalShift()

    # Initial affine transform
    reg.SetInitialTransform(initial_affine, inPlace=False)
    final_affine = reg.Execute(fixed, moving)

    #print("Final metric:", reg.GetMetricValue())
    #print("Optimizer stop condition:", reg.GetOptimizerStopConditionDescription())

    # 7. Resample moving into fixed space with final affine
    resampled = sitk.Resample(
        moving,
        fixed,
        final_affine,
        sitk.sitkLinear,
        0.0,
        sitk.sitkFloat32,
    )

    # Back to numpy in (X, Y, Z)
    resampled_zyx = sitk.GetArrayFromImage(resampled)       # (Z, Y, X)
    resampled_xyz = np.transpose(resampled_zyx, (2, 1, 0))  # (X, Y, Z)

    return resampled_xyz, final_affine
