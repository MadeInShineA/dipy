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

    # Brain mask packages
    from dipy.segment.mask import median_otsu
    from dipy.core.histeq import histeq
    return (
        get_fnames,
        gradient_table,
        load_nifti,
        median_otsu,
        mo,
        np,
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
    #fraw, fbval, fbvec = get_fnames(name="stanford_hardi")
    return fbval, fbvec, fraw


@app.cell
def _(fbval, fbvec, fraw, gradient_table, load_nifti, np, read_bvals_bvecs):
    data, affine, voxel_size = load_nifti(fraw, return_voxsize=True)
    bvals, bvecs = read_bvals_bvecs(fbval, fbvec)
    bvecs[1:] = bvecs[1:] / np.sqrt(np.sum(bvecs[1:] * bvecs[1:], axis=1))[:, None]
    gtab = gradient_table(bvals=bvals, bvecs=bvecs)
    print(f"data.shape {data.shape}")
    return bvals, bvecs, data, gtab


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Visualize the data
    """)
    return


@app.cell
def _():
    import matplotlib.pyplot as plt

    def plot_slice(data, bval, bvec, volume_idx, z_slice=None):
        """
        Plot a single axial slice from a 3D diffusion MRI volume.

        Parameters:
        - data: 3D array of the volume (X, Y, Z)
        - bval: b-value (in s/mm²) of the volume, shown in the title
        - bvec: 3-element b-vector (diffusion gradient direction), shown in the title
        - volume_idx: index of the volume in the original 4D dataset (used for labeling)
        - z_slice: axial slice index along the Z dimension; if None, uses the central slice
        """
        if z_slice is None:
            z_slice = data.shape[2] // 2

        vol = data[:, :, z_slice]

        fig, ax = plt.subplots(figsize=(10, 10))

        im = ax.imshow(vol, cmap='gray', origin='lower')
        ax.set_title(f'Volume {volume_idx} | bval ={bval} s/mm^2 | bvec = {bvec}')
        ax.axis('off')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        plt.tight_layout()
        plt.show()
    return plot_slice, plt


@app.cell
def _(bvals, bvecs, data, np, plot_slice):
    # Plot 5 random slices
    np.random.seed(42)
    random_slices= np.random.choice(data.shape[-1], size=5, replace=False)

    for random_idx in random_slices:
        plot_slice(data[...,random_idx], bvals[random_idx], bvecs[random_idx], random_idx)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Split the data and gtab into train and test sets
     - Note that B0 volumes are excluded from train and test sets
    """)
    return


@app.cell
def _(gtab, np):
    b0_indices = np.where(gtab.b0s_mask)[0]
    non_b0_indices = np.where(~gtab.b0s_mask)[0]

    n_test = 2
    n_train = len(non_b0_indices) - n_test

    # Shuffle only the non-b0 indices
    permuted_non_b0 = np.random.permutation(non_b0_indices)
    train_idx = permuted_non_b0[:n_train]
    test_idx = permuted_non_b0[n_train:]

    print(f"Number of train gradients: {len(train_idx)}")
    print(f"Number of test gradients: {len(test_idx)}")
    print(f"Number of b=0 volumes: {len(b0_indices)}")
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
    return test_bvals, test_bvecs, test_data, test_gtab, train_data, train_gtab


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Compute a brain mask to make fit computation faster
    """)
    return


@app.cell
def _(data, median_otsu):
    masked_data, mask = median_otsu(data, vol_idx=[0])
    return (mask,)


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
def _(mask, test_gtab, train_data, train_gtab):
    import dipy.reconst.gqi as gqi
    import time

    methods = ["standard", "gqi2"]
    method_predicted_data = {method: {} for method in methods}

    for method in methods:
        model = gqi.GeneralizedQSamplingModel(train_gtab, method=method, sampling_length=0.9)
        fit = model.fit(train_data, mask=mask)

        predicted_data = fit.predict(test_gtab)

        method_predicted_data[method]["predicted_data"] = predicted_data
    return gqi, method_predicted_data


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Plot the predicted results
    """)
    return


@app.cell
def _(np):
    from scipy.stats import pearsonr
    from scipy.ndimage import generic_filter

    def local_correlation(real, pred, window_size=9):
        """
        Compute a local Pearson correlation map between `real` and `pred`.
        """
        if window_size % 2 == 0:
            raise ValueError("window_size must be odd.")

        # Ensure float dtype to support NaN
        real = real.astype(np.float32)
        pred = pred.astype(np.float32)

        pad = window_size // 2
        real_padded = np.pad(real, pad, mode='constant', constant_values=np.nan)
        pred_padded = np.pad(pred, pad, mode='constant', constant_values=np.nan)

        corr_map = np.full_like(real, np.nan, dtype=np.float32)
        h, w = real.shape

        for i in range(h):
            for j in range(w):
                real_patch = real_padded[i:i+window_size, j:j+window_size].flatten()
                pred_patch = pred_padded[i:i+window_size, j:j+window_size].flatten()

                valid = ~(np.isnan(real_patch) | np.isnan(pred_patch))
                if valid.sum() < 2:
                    corr_map[i, j] = np.nan
                else:
                    r, _ = pearsonr(real_patch[valid], pred_patch[valid])
                    corr_map[i, j] = r
        return corr_map
    return (local_correlation,)


@app.cell
def _(local_correlation, plt):
    def plot_all_methods_comparison(
        test_data, method_predicted_data, vol_idx, test_bval, test_bvec, z_slice=None, window_size=9
    ):
        """
        Plot real data, predictions from all methods, and local Pearson correlation maps.
        Each method gets its own row with three panels: [Real | Predicted | Local Correlation].

        Parameters:
        - test_data: 3D array of ground-truth test volume (X, Y, Z)
        - method_predicted_data: 
            dict mapping method names to prediction containers;
            each value must contain a key "predicted_data" with a 4D array (X, Y, Z, N_vols)
        - vol_idx: index of the test volume to visualize (selects the 4th dimension of predictions)
        - test_bval: b-value (in s/mm²) of the selected volume, shown in the plot title
        - test_bvec: 3-element b-vector (gradient direction) of the selected volume, shown in the title
        - z_slice: axial slice index along the Z dimension; if None, uses the central slice
        - window_size: size of the square window for local correlation (must be odd; default: 9)
        """

        if z_slice is None:
            z_slice = test_data.shape[2] // 2

        real_vol = test_data[:, :, z_slice]
        methods = list(method_predicted_data.keys())
        n_methods = len(methods)

        fig, axes = plt.subplots(n_methods, 3, figsize=(13, 3.8 * n_methods))

        if n_methods == 1:
            axes = axes[None, :]

        for i, method in enumerate(methods):
            pred_vol = method_predicted_data[method]["predicted_data"][
                :, :, z_slice, vol_idx
            ]

            # Compute local correlation map
            real_pred_corr_map = local_correlation(
                real_vol, pred_vol, window_size=window_size
            )

            # Real
            im_real = axes[i, 0].imshow(real_vol, cmap="gray", origin="lower")
            if i == 0:
                axes[i, 0].set_title("Real data", fontsize=12)
            axes[i, 0].axis("off")
            fig.colorbar(im_real, ax=axes[i, 0], fraction=0.046, pad=0.04)

            # Predicted
            im_pred = axes[i, 1].imshow(pred_vol, cmap="gray", origin="lower")
            axes[i, 1].set_title(f"{method.upper()} prediction", fontsize=12)
            axes[i, 1].axis("off")
            fig.colorbar(im_pred, ax=axes[i, 1], fraction=0.046, pad=0.04)

            # Local Correlation (real vs prediction)
            im_corr = axes[i, 2].imshow(
                real_pred_corr_map, cmap="RdBu_r", origin="lower", vmin=-1, vmax=1
            )
            axes[i, 2].set_title(
                "Local Pearson Correlation\n(real vs prediction)", fontsize=12
            )
            axes[i, 2].axis("off")
            fig.colorbar(im_corr, ax=axes[i, 2], fraction=0.046, pad=0.04)

        fig.suptitle(
            f"Test Volume {vol_idx} | Z Slice {z_slice} | "
            f"Corr window size {window_size} | bval = {test_bval:.0f} s/mm^2 | bvec = {test_bvec}",
            fontsize=14,
            y=0.98,
        )
        plt.tight_layout(rect=[0, 0, 1, 0.96], h_pad=4.0)
        plt.show()
    return (plot_all_methods_comparison,)


@app.cell
def _(
    method_predicted_data,
    plot_all_methods_comparison,
    test_bvals,
    test_bvecs,
    test_data,
    test_idx,
):
    for i in range(len(test_idx)):
        plot_all_methods_comparison(test_data[..., i], method_predicted_data, i, test_bvals[i], test_bvecs[i], window_size=9)
    return


@app.cell
def _(mo):
    mo.md(r"""
    #Perform k fold cross validations with already existing models (see [existing tutorial](https://docs.dipy.org/stable/examples_built/reconstruction/kfold_xval.html))
    - DTI
    - CSD
    - GQI
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Select a couple of voxels to perform comparisons on.

    One lies in the corpus callosum (cc), while the other is in the centrum semiovale (cso), a part of the brain known to contain multiple crossing white matter fiber populations.
    """)
    return


@app.cell
def _(data):
    cc_vox = data[40, 70, 38]
    cso_vox = data[30, 76, 38]
    return cc_vox, cso_vox


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Initialize each model
    """)
    return


@app.cell
def _(data, gqi, gtab):
    import dipy.reconst.csdeconv as csd
    import dipy.reconst.dti as dti 

    dti_model = dti.TensorModel(gtab)
    response, ratio = csd.auto_response_ssst(gtab, data, roi_radii=10, fa_thr=0.7)
    csd_model = csd.ConstrainedSphericalDeconvModel(gtab, response)
    gqi_model = gqi.GeneralizedQSamplingModel(gtab, method="standard", sampling_length=0.9)
    return csd_model, dti_model, gqi_model, response


@app.cell
def _(mo):
    mo.md(r"""
    ## Perform cross-validation for each kind of model

    Note that we use 2-fold cross-validation, which means that in each iteration, the model will be fit to half of the data, and used to predict the other half.
    """)
    return


@app.cell
def _(cc_vox, csd_model, cso_vox, dti_model, gqi_model, np, response):
    import dipy.reconst.cross_validation as xval

    rng = np.random.default_rng(2014)

    dti_cc = xval.kfold_xval(dti_model, cc_vox, 2, rng=rng)
    csd_cc = xval.kfold_xval(csd_model, cc_vox, 2, response, rng=rng)
    gqi_cc = xval.kfold_xval(gqi_model, cc_vox, 2, rng=rng)

    dti_cso = xval.kfold_xval(dti_model, cso_vox, 2, rng=rng)
    csd_cso = xval.kfold_xval(csd_model, cso_vox, 2, response, rng=rng)
    gqi_cso = xval.kfold_xval(gqi_model, cso_vox, 2, rng=rng)
    return csd_cc, csd_cso, dti_cc, dti_cso, gqi_cc, gqi_cso


@app.cell
def _(mo):
    mo.md(r"""
    ## Plot results
    """)
    return


@app.cell
def _(
    cc_vox,
    csd_cc,
    csd_cso,
    cso_vox,
    dti_cc,
    dti_cso,
    gqi_cc,
    gqi_cso,
    gtab,
    plt,
):
    def _():
        fig, ax = plt.subplots(1, 3)
        fig.set_size_inches([12, 6])
        ax[0].plot(
            cc_vox[gtab.b0s_mask == 0],
            dti_cc[gtab.b0s_mask == 0],
            "o",
            color="b",
            label="DTI in CC",
        )
        ax[0].plot(
            cc_vox[gtab.b0s_mask == 0],
            csd_cc[gtab.b0s_mask == 0],
            "o",
            color="r",
            label="CSD in CC",
        )
        ax[1].plot(
            cso_vox[gtab.b0s_mask == 0],
            dti_cso[gtab.b0s_mask == 0],
            "o",
            color="b",
            label="DTI in CSO",
        )
        ax[1].plot(
            cso_vox[gtab.b0s_mask == 0],
            csd_cso[gtab.b0s_mask == 0],
            "o",
            color="r",
            label="CSD in CSO",
        )
        ax[2].plot(
                cc_vox[gtab.b0s_mask == 0],
                gqi_cc[gtab.b0s_mask == 0],
                "o",
                color="b",
                label="GQI in CC",
            )
        ax[2].plot(
            cc_vox[gtab.b0s_mask == 0],
            gqi_cso[gtab.b0s_mask == 0],
            "o",
            color="r",
            label="GQI in CSO",
        )
        ax[0].legend(loc="upper left")
        ax[1].legend(loc="upper left")
        ax[2].legend(loc="upper left")
        for this_ax in ax:
            this_ax.set_xlabel("Data (relative to S0)")
            this_ax.set_ylabel("Model prediction (relative to S0)")
        plt.show()


    _()
    return


@app.cell
def _(
    cc_vox,
    csd_cc,
    csd_cso,
    cso_vox,
    dti_cc,
    dti_cso,
    gqi_cc,
    gqi_cso,
    gtab,
):
    import scipy.stats as stats

    cc_dti_r2 = (
        stats.pearsonr(cc_vox[gtab.b0s_mask == 0], dti_cc[gtab.b0s_mask == 0])[0] ** 2
    )
    cc_csd_r2 = (
        stats.pearsonr(cc_vox[gtab.b0s_mask == 0], csd_cc[gtab.b0s_mask == 0])[0] ** 2
    )
    cc_gqi_r2 = (
        stats.pearsonr(cc_vox[gtab.b0s_mask == 0], gqi_cc[gtab.b0s_mask == 0])[0] ** 2

    )
    cso_dti_r2 = (
        stats.pearsonr(cso_vox[gtab.b0s_mask == 0], dti_cso[gtab.b0s_mask == 0])[0] ** 2
    )
    cso_csd_r2 = (
        stats.pearsonr(cso_vox[gtab.b0s_mask == 0], csd_cso[gtab.b0s_mask == 0])[0] ** 2
    )
    cso_gqi_r2 = (
        stats.pearsonr(cso_vox[gtab.b0s_mask == 0], gqi_cso[gtab.b0s_mask == 0])[0] ** 2
    )

    print(
        "Corpus callosum\n"
        f"DTI R2 : {cc_dti_r2}\n"
        f"CSD R2 : {cc_csd_r2}\n"
        f"GQI R2 : {cc_gqi_r2}\n"

        "\n"
        "Centrum Semiovale\n"
        f"DTI R2 : {cso_dti_r2}\n"
        f"CSD R2 : {cso_csd_r2}\n"
        f"GQI R2 : {cso_gqi_r2}\n"

    )
    return


if __name__ == "__main__":
    app.run()
