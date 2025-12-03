import marimo

__generated_with = "0.18.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    from dipy.data import get_fnames
    from dipy.io.image import load_nifti
    from dipy.io.gradients import read_bvals_bvecs
    import numpy as np
    from dipy.core.gradients import gradient_table
    from dipy.reconst.gqi import GeneralizedQSamplingModel
    import matplotlib.pyplot as plt
    return (
        GeneralizedQSamplingModel,
        get_fnames,
        gradient_table,
        load_nifti,
        mo,
        np,
        plt,
        read_bvals_bvecs,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Predict unseen data notebook example
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Load the data
    """)
    return


@app.cell
def _(get_fnames):
    fraw, fbval, fbvec = get_fnames(name="taiwan_ntu_dsi")
    return fbval, fbvec, fraw


@app.cell
def _(fbval, fbvec, fraw, gradient_table, load_nifti, np, read_bvals_bvecs):
    data, affine, voxel_size = load_nifti(fraw, return_voxsize=True)
    bvals, bvecs = read_bvals_bvecs(fbval, fbvec)
    bvecs[1:] = bvecs[1:] / np.sqrt(np.sum(bvecs[1:] * bvecs[1:], axis=1))[:, None]
    gtab = gradient_table(bvals, bvecs=bvecs)
    print(f"data.shape {data.shape}")
    return affine, bvals, bvecs, data


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Visualize the data
    """)
    return


@app.cell
def _(plt):
    def plot_slice(data, bvals, vol_idx=0, z_slice=None, title_prefix="Volume"):
        """
        Plot a single 2D slice from a 4D diffusion volume.

        Parameters:
        - data: 4D array of shape (X, Y, Z, N)
        - bvals: 1D array of b-values (length N)
        - vol_idx: which volume (4th dim) to show
        - z_slice: axial slice index; if None, use middle slice
        - title_prefix: label for the plot title
        """
        if z_slice is None:
            z_slice = data.shape[2] // 2  # Use data, not test_data (more general)

        vol = data[:, :, z_slice, vol_idx]

        fig, ax = plt.subplots(figsize=(6, 5))

        im = ax.imshow(vol, cmap='gray', origin='lower')
        ax.set_title(f'{title_prefix} | Volume {vol_idx} (b={bvals[vol_idx]:.0f})')
        ax.axis('off')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        plt.tight_layout()
        plt.show()
    return (plot_slice,)


@app.cell
def _(bvals, data, np, plot_slice):
    # Plot 10 random slices
    np.random.seed(42)
    random_slices = np.random.choice(data.shape[-1], size=10, replace=False)

    for idx in random_slices:
        plot_slice(data, bvals, idx)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Initialize the Qball model
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Split the data and gtab into train and test sets
    """)
    return


@app.cell
def _(data, np):

    N = data.shape[-1]
    indices = np.random.permutation(N)

    test_proportion = 0.01

    split = int((1 - test_proportion) * N)
    train_idx, test_idx = indices[:split], indices[split:]

    print(f"Number of train gradients: {len(train_idx)}")
    print(f"Number of test gradients: {len(test_idx)}")
    return test_idx, train_idx


@app.cell
def _(bvals, bvecs, data, gradient_table, test_idx, train_idx):
    # Split the 4D data (along last axis)
    train_data = data[..., train_idx]
    test_data  = data[..., test_idx]

    # Split bvals and bvecs
    train_bvals = bvals[train_idx]
    train_bvecs = bvecs[train_idx]

    test_bvals = bvals[test_idx]
    test_bvecs = bvecs[test_idx]

    train_gtab = gradient_table(bvals=train_bvals, bvecs=train_bvecs)
    test_gtab = gradient_table(bvals=test_bvals, bvecs=test_bvecs)

    print(f"Train data shape {train_data.shape}")
    print(f"Test data shape {test_data.shape}")

    print(f"Number of train voxels: {train_data.size}")
    print(f"Number of test voxels: {test_data.size}")
    return test_bvals, test_data, test_gtab, train_data, train_gtab


@app.cell
def _(plot_slice, test_bvals, test_data, test_idx):
    # Plot the test slices
    for _i in range(len(test_idx)):
        print(_i)
        plot_slice(test_data, test_bvals, _i)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## For each of the GQI method (standard, gqi2):
    - Create the gqi model
    - Fit the train data to the model
    - Predict the unseen test data
    """)
    return


@app.cell
def _(GeneralizedQSamplingModel, test_gtab, train_data, train_gtab):
    methods = ["standard"]

    method_predicted_data = {}
    for method in methods:
        gqmodel = GeneralizedQSamplingModel(train_gtab, method=method, sampling_length=0.9)
        fit = gqmodel.fit(train_data)
        predicted_data = fit.predict(test_gtab)
        method_predicted_data[method] = predicted_data
    return (method_predicted_data,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Plot the predicted results
    """)
    return


@app.cell
def _(np, plt):
    def plot_all_methods_comparison(test_data, test_bvals, method_predicted_data, 
                                   vol_idx, z_slice=None):
        """
        Plot real data, predictions from all methods, and absolute error maps.
        Each method gets its own row: [Real | Predicted | Error]
        Now includes a colorbar next to each plot.

        Parameters:
        - test_data: ndarray of shape (H, W, D, V) — ground truth
        - method_predicted_data: dict {method_name: prediction_array}
        - vol_idx: which volume (4th dim) to visualize
        - z_slice: which axial slice (3rd dim); defaults to middle
        - test_bvals: optional b-values for title annotation
        """
        if z_slice is None:
            z_slice = test_data.shape[2] // 2

        real_vol = test_data[:, :, z_slice, vol_idx]
        methods = list(method_predicted_data.keys())
        n_methods = len(methods)

        # Adjust figure size to accommodate colorbars
        fig, axes = plt.subplots(n_methods, 3, figsize=(12, 3.8 * n_methods))

        if n_methods == 1:
            axes = axes[None, :]  # make it (1, 3)

        for i, method in enumerate(methods):
            pred_vol = method_predicted_data[method][:, :, z_slice, vol_idx]
            error = np.abs(real_vol - pred_vol)

            # Real
            im_real = axes[i, 0].imshow(real_vol, cmap='gray', origin='lower')
            if i == 0:
                axes[i, 0].set_title('Real data', fontsize=12)
            axes[i, 0].axis('off')
            fig.colorbar(im_real, ax=axes[i, 0], fraction=0.046, pad=0.04)

            # Predicted
            im_pred = axes[i, 1].imshow(pred_vol, cmap='gray', origin='lower')
            axes[i, 1].set_title(f'{method.upper()} prediction', fontsize=12)
            axes[i, 1].axis('off')
            fig.colorbar(im_pred, ax=axes[i, 1], fraction=0.046, pad=0.04)

            # Error
            im_error = axes[i, 2].imshow(error, cmap='hot', origin='lower')
            axes[i, 2].set_title('Absolute Error\n(real - prediction)', fontsize=12)
            axes[i, 2].axis('off')
            fig.colorbar(im_error, ax=axes[i, 2], fraction=0.046, pad=0.04)

        # Suptitle
        fig.suptitle(f'Test Volume {vol_idx} (b = {test_bvals[vol_idx]:.0f})', fontsize=14, y=0.98)

        plt.tight_layout(rect=[0, 0, 1, 0.96], h_pad=4.0)
        plt.show()
    return (plot_all_methods_comparison,)


@app.cell
def _(
    method_predicted_data,
    plot_all_methods_comparison,
    test_bvals,
    test_data,
    test_idx,
):
    for i in range(len(test_idx)):
        plot_all_methods_comparison(test_data, test_bvals, method_predicted_data, vol_idx=i)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Add image registration (To check)
    """)
    return


@app.cell
def _(affine, method_predicted_data, np, test_data):
    from dipy.align import affine_registration

    # For each test volume (assuming same affine)
    registered_predictions = {}
    for _method, pred in method_predicted_data.items():
        registered = []
        for _i in range(pred.shape[-1]):
            reg_pred, _ = affine_registration(
                pred[..., _i], test_data[..., _i],
                moving_affine=affine, static_affine=affine,
                nbins=32, metric="MI",
                pipeline=["center_of_mass", "translation", "rigid", "affine"]
            )
            registered.append(reg_pred)
        registered_predictions[_method] = np.stack(registered, axis=-1)
    return (registered_predictions,)


@app.cell
def _(
    plot_all_methods_comparison,
    registered_predictions,
    test_bvals,
    test_data,
    test_idx,
):
    for _i in range(len(test_idx)):
        plot_all_methods_comparison(test_data, test_bvals, registered_predictions, vol_idx=_i)
    return


if __name__ == "__main__":
    app.run()
