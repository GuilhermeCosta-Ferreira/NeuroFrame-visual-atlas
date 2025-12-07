# ================================================================
# 0. Section: Imports
# ================================================================
import SimpleITK as sitk
import numpy as np



# ================================================================
# 1. Section: Rigid Registration of Two Brain MRIs
# ================================================================
def rigid_registration(reference_brain_path: str, moving_brain: str | np.ndarray, transform: sitk.Transform = None) -> tuple[np.ndarray, sitk.Transform]:
    """Perform a rigid registration of a moving brain MRI to a reference brain MRI."""

    fixed  = sitk.ReadImage(reference_brain_path, sitk.sitkFloat32)

    if(isinstance(moving_brain, np.ndarray)): moving = sitk.GetImageFromArray(moving_brain)
    else: moving = sitk.ReadImage(moving_brain, sitk.sitkFloat32)

    moving.SetSpacing(fixed.GetSpacing())

    if(transform is not None): initial_tx = transform
    else:
        initial_tx = sitk.CenteredTransformInitializer(
            fixed,
            moving,
            sitk.Euler3DTransform(),
            sitk.CenteredTransformInitializerFilter.MOMENTS,
        )

    reg = sitk.ImageRegistrationMethod()

    # Metric
    reg.SetMetricAsMattesMutualInformation()
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

    # Initial transform (rigid: rotation + translation only)
    reg.SetInitialTransform(initial_tx, inPlace=False)

    final_tx = reg.Execute(fixed, moving)

    print("Final metric:", reg.GetMetricValue())
    print("Optimizer stop condition:", reg.GetOptimizerStopConditionDescription())

    # ---- resample moving into fixed space ----
    resampled = sitk.Resample(
        moving,
        fixed,
        final_tx,
        sitk.sitkLinear,
        0.0,
        moving.GetPixelID(),
    )

    resampled = sitk.GetArrayFromImage(resampled)
    return np.transpose(resampled, (2, 1, 0)), final_tx
