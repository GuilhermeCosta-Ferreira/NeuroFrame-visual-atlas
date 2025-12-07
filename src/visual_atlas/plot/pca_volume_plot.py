# ================================================================
# 0. Section: Imports
# ================================================================
import matplotlib.pyplot as plt
import numpy as np

from ..reference_align import get_brain_pca_components
from .plot_utils import get_main_slices



# ================================================================
# 1. Section: PCA Volume Plotting
# ================================================================
def plot_pca_orientations(volume: np.ndarray, **kwargs):
    """Plot the principal components of a 3D volume on its main slices.

    This function computes the Principal Component Analysis (PCA) of a given 3D
    volume and visualizes the resulting principal axes. It generates a figure
    with three subplots, each showing a central slice from one of the three
    primary orientations. The PCA vectors are overlaid on these slices.

    Parameters
    ----------
    volume : np.ndarray
        The 3D numpy array representing the volume data (e.g., a brain scan).
    **kwargs : dict
        Additional keyword arguments to be passed to the internal plotting
        function `_plot_slice_pca`, which handles the drawing of PCA vectors
        on each slice.

    Returns
    -------
    None
        This function displays a matplotlib plot and does not return any value.

    See Also
    --------
    get_brain_pca_components : Computes the PCA components for the volume.
    get_main_slices : Extracts the central slices from the volume.

    Examples
    --------
    >>> import numpy as np
    >>> # Create a dummy volume for demonstration
    >>> dummy_volume = np.zeros((100, 100, 100))
    >>> dummy_volume[40:60, 40:60, 40:60] = 1
    >>> plot_pca_orientations(dummy_volume)

    Notes
    -----
    - The function creates and displays a new matplotlib figure.
    - It assumes the volume has three primary orientations and visualizes the
      middle slice for each.
    """
    
    # Compute PCA components for the volume
    volume_pca = get_brain_pca_components(volume)

    # Start the plotting
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Get the main slices and plot PCA on each
    main_slices = get_main_slices(volume)
    for idx, (orientation, slice) in enumerate(main_slices.items()):
        target_ax = axes[idx]
        _plot_slice_pca(target_ax, slice, volume_pca, orientation.lower(), **kwargs)

    # Add overall title and show plot
    fig.suptitle("PCA Components on Middle Slices")
    plt.tight_layout()
    plt.show()


# ──────────────────────────────────────────────────────
# 1.1 Subsection: Individual PCA Slice Plotting
# ──────────────────────────────────────────────────────
def _plot_slice_pca(axes: plt.Axes, slice: np.ndarray, volume_pca: np.ndarray, orientation: str, **kwargs):
    """Plot a single slice with PCA components overlaid as arrows."""
    scale = kwargs.get('scale', 20)
    colors = kwargs.get('colors', ['#B14E4E', '#4EB14E', '#4E4EB1'])

    axes.imshow(slice, cmap='gray')
    center_x, center_y = slice.shape[1] // 2, slice.shape[0] // 2
    x, y = _get_vector_indecs(orientation)

    for i in range(3):
        axes.arrow(center_x, center_y,
                    volume_pca[x, i] * scale,
                    volume_pca[y, i] * scale,
                    head_width=5, head_length=5, fc=colors[i], ec=colors[i])
    axes.set_title(f"{orientation.title()} View")
    axes.axis('off')

# ››››››››››››››››››››››››››››››››››››››››››››››››
# 1.1.1 Sub-subsection: Helper Functions
# ›››››››››››››››››››››››››››››››››››››››››››››››››
def _get_vector_indecs(orientation: str):
    """Get the indices of the PCA components to plot based on the slice orientation."""
    orientation = orientation.lower()
    if orientation == 'axial': return 2, 1
    elif orientation == 'coronal': return 2, 0
    elif orientation == 'sagittal': return 1, 0 
    else: raise ValueError("Orientation must be 'axial', 'coronal', or 'sagittal'.")