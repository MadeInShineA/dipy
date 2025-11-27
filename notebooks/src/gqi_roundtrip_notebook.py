import marimo

__generated_with = "0.18.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from dipy.data import dsi_voxels
    from dipy.reconst.gqi import GeneralizedQSamplingModel
    return GeneralizedQSamplingModel, dsi_voxels, mo, np, plt


@app.cell
def _(mo):
    mo.md("""
    # GQI Round-Trip Prediction Notebook

    This notebook demonstrates **round-trip consistency** in GQI reconstruction:
    - Fit GQI model on diffusion data
    - Predict signals using the **same gradients** used for fitting
    - Compare original vs predicted signals to validate reconstruction accuracy

    **Key Concepts:**
    - Round-trip consistency tests if the model can accurately reconstruct training data
    - High correlation indicates good model fit
    - Tests both single-voxel and multi-voxel scenarios
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 1. Load Real DSI Data

    Using the same small_101D DSI dataset as in the tests
    """)
    return


@app.cell
def _(dsi_voxels, np):
    # Load real DSI data
    data, gtab = dsi_voxels()
    print(f"Data shape: {data.shape}")
    print(f"Number of gradients: {len(gtab.bvals)}")
    print(f"b-values range: [{gtab.bvals.min()}, {gtab.bvals.max()}]")
    print(f"Number of b=0 images: {np.sum(gtab.b0s_mask)}")
    return data, gtab


@app.cell
def _(mo):
    mo.md("""
    ## 2. Single Voxel Round-Trip Test

    Test reconstruction accuracy on a single voxel
    """)
    return


@app.cell
def _(GeneralizedQSamplingModel, data, gtab, np):
    # Select a voxel with good signal
    voxel_coord = (0, 0, 0)
    voxel_data = data[voxel_coord]

    print(f"Selected voxel at {voxel_coord}")
    print(f"Signal range: [{voxel_data.min():.3f}, {voxel_data.max():.3f}]")
    print(f"Mean signal: {voxel_data.mean():.3f}")

    # Test both methods
    methods = ["standard", "gqi2"]
    results = {}

    for _method in methods:
        print(f"\n--- Testing {_method} method ---")

        # Fit GQI model
        _gq = GeneralizedQSamplingModel(gtab, method=_method, sampling_length=1.2)
        voxel_fit = _gq.fit(voxel_data)

        # Predict on same gradients (round-trip)
        _voxel_predicted = voxel_fit.predict(gtab)

        # Calculate metrics
        _correlation = np.corrcoef(voxel_data, _voxel_predicted)[0, 1]
        _mae = np.mean(np.abs(voxel_data - _voxel_predicted))

        print(f"Correlation: {_correlation:.4f}")
        print(f"Mean Absolute Error: {_mae:.4f}")
        print(
            f"Predicted range: [{_voxel_predicted.min():.3f}, {_voxel_predicted.max():.3f}]"
        )

        results[_method] = {
            "original": voxel_data,
            "predicted": _voxel_predicted,
            "correlation": _correlation,
            "mae": _mae,
        }
    return results, voxel_coord


@app.cell
def _(mo):
    mo.md("""
    ## 3. Single Voxel Visualization
    """)
    return


@app.cell
def _(np, plt, results, voxel_coord):
    def plot_voxel_reconstruction(results, voxel_coord, figsize=(14, 9)):
        methods = ["standard", "gqi2"]
        method_labels = ["Standard GQI", "GQI2"]
        colors = ["#1f77b4", "#ff7f0e"]

        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle(
            f"Voxel {voxel_coord} Signal Reconstruction: Original vs. Predicted",
            fontsize=14,
            weight='bold'
        )

        x = np.arange(len(results["standard"]["original"]))

        for row, (method, label, color) in enumerate(zip(methods, method_labels, colors)):
            orig = np.array(results[method]["original"])
            pred = np.array(results[method]["predicted"])
            corr = results[method]["correlation"]
            mae = results[method]["mae"]

            # --- Left: Scatter plot (0 to max) ---
            ax_scatter = axes[row, 0]
            ax_scatter.scatter(orig, pred, alpha=0.7, s=25, color=color, edgecolor='k', linewidth=0.3)

            # Compute shared axis limit: from 0 to max of both signals
            max_val = max(orig.max(), pred.max())
            ax_scatter.set_xlim(0, max_val)
            ax_scatter.set_ylim(0, max_val)

            # Identity line (only within [0, max_val])
            ax_scatter.plot([0, max_val], [0, max_val], 'k--', lw=1.5, label="Perfect fit")

            # Linear fit (optional but useful)
            slope, intercept = np.polyfit(orig, pred, 1)
            fit_x = np.array([0, max_val])
            fit_y = slope * fit_x + intercept
            ax_scatter.plot(fit_x, fit_y, color='red', lw=1.8, label=f"Fit: y={slope:.2f}x+{intercept:.2f}")

            ax_scatter.set_xlabel("Original Signal")
            ax_scatter.set_ylabel("Predicted Signal")
            ax_scatter.set_title(f"{label}\nPearson r = {corr:.4f}")
            ax_scatter.legend(fontsize=8)
            ax_scatter.grid(True, alpha=0.3)
            ax_scatter.set_aspect('equal', adjustable='box')  # square axes

            # --- Right: Signal + residuals ---
            ax_signal = axes[row, 1]
            ax_signal.plot(x, orig, 'o-', color='black', label="Original", markersize=4, linewidth=1.2)
            ax_signal.plot(x, pred, 'o--', color=color, label="Predicted", markersize=4, linewidth=1.2)

            # Residuals on twin axis
            ax_resid = ax_signal.twinx()
            residuals = pred - orig
            ax_resid.vlines(x, 0, residuals, colors='gray', linestyles=':', alpha=0.6, linewidth=1)
            ax_resid.axhline(0, color='gray', linewidth=0.8)
            ax_resid.set_ylabel("Residual (Pred – Orig)", color='gray')
            ax_resid.tick_params(axis='y', labelcolor='gray')

            ax_signal.set_xlabel("Gradient Index")
            ax_signal.set_ylabel("Signal Intensity")
            ax_signal.set_title(f"{label}\nMAE = {mae:.4f}")
            ax_signal.legend()
            ax_signal.grid(True, alpha=0.3)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        return fig

    plot_voxel_reconstruction(results, voxel_coord)

    return


@app.cell
def _(mo):
    mo.md("""
    ## 4. Multi-Voxel Round-Trip Test

    Test reconstruction accuracy across multiple voxels
    """)
    return


@app.cell
def _(GeneralizedQSamplingModel, data, gtab, np):
    # Test multi-voxel round-trip for both methods
    multi_results = {}

    for _method in ["standard", "gqi2"]:
        print(f"\n=== Multi-voxel {_method} method ===")

        # Fit on entire 3D dataset
        _gq = GeneralizedQSamplingModel(gtab, method=_method, sampling_length=1.2)
        multi_fit = _gq.fit(data)

        # Predict on same gradients
        multi_predicted = multi_fit.predict(gtab)

        # Calculate voxel-wise correlations
        correlations = []
        total_voxels = data.shape[0] * data.shape[1] * data.shape[2]

        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                for k in range(data.shape[2]):
                    original_voxel = data[i, j, k]
                    predicted_voxel = multi_predicted[i, j, k]

                    # Skip voxels with no signal
                    if np.sum(original_voxel) == 0:
                        continue

                    correlation = np.corrcoef(original_voxel, predicted_voxel)[0, 1]
                    correlations.append(correlation)

        _correlations = np.array(correlations)

        print(f"Total voxels: {total_voxels}")
        print(f"Average correlation: {np.mean(correlations):.4f}")
        print(f"Min correlation: {np.min(_correlations):.4f}")
        print(f"Max correlation: {np.max(_correlations):.4f}")
        print(f"Std correlation: {np.std(_correlations):.4f}")

        # Global metrics
        overall_correlation = np.corrcoef(data.flatten(), multi_predicted.flatten())[
            0, 1
        ]
        overall_mae = np.mean(np.abs(data - multi_predicted))

        print(f"Overall correlation: {overall_correlation:.4f}")
        print(f"Overall MAE: {overall_mae:.4f}")

        multi_results[_method] = {
            "predicted": multi_predicted,
            "correlations": correlations,
            "avg_correlation": np.mean(correlations),
            "overall_correlation": overall_correlation,
            "overall_mae": overall_mae,
        }
    return (multi_results,)


@app.cell
def _(mo):
    mo.md("""
    ## 5. Multi-Voxel Results Visualization
    """)
    return


@app.cell
def _(data, multi_results, np, plt):
    def plot_multi_voxel_results(multi_results, data_shape, figsize=(14, 10)):
        methods = ["standard", "gqi2"]
        method_labels = ["Standard GQI", "GQI2"]
        colors = ["#1f77b4", "#ff7f0e"]

        # Reconstruct 3D correlation maps for spatial visualization
        corr_maps = {}
        for method in methods:
            correlations_flat = np.array(multi_results[method]["correlations"])
            corr_map = np.full(data_shape[:3], np.nan)  # (X, Y, Z)

            # Refill in original order
            idx = 0
            for i in range(data_shape[0]):
                for j in range(data_shape[1]):
                    for k in range(data_shape[2]):
                        if np.sum(data[i, j, k]) > 0:
                            corr_map[i, j, k] = correlations_flat[idx]
                            idx += 1
            corr_maps[method] = corr_map

        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle("Voxel-wise Signal Prediction Accuracy: Standard vs. GQI2", fontsize=14, weight='bold')

        for row, (method, label, color) in enumerate(zip(methods, method_labels, colors)):
            corr_flat = np.array(multi_results[method]["correlations"])
            corr_map = corr_maps[method]

            # 1. Histogram (left column)
            ax_hist = axes[row, 0]
            ax_hist.hist(corr_flat, bins=30, alpha=0.7, color=color, edgecolor='black', density=True)
            mean_corr = np.mean(corr_flat)
            ax_hist.axvline(mean_corr, color='red', linestyle='--', linewidth=2,
                            label=f"Mean: {mean_corr:.4f}")
            ax_hist.set_xlabel("Voxel-wise Pearson Correlation")
            ax_hist.set_ylabel("Density")
            ax_hist.set_title(f"{label}: Correlation Distribution")
            ax_hist.legend()
            ax_hist.grid(True, alpha=0.3)

            # 2. Central slice of correlation map (right column)
            ax_slice = axes[row, 1]
            # Choose middle slice along the largest dimension for visibility
            mid_i, mid_j, mid_k = np.array(data_shape[:3]) // 2
            # Prefer axial (k) if reasonable size
            if data_shape[2] > 1:
                slice_data = corr_map[:, :, mid_k]
                title = f"{label}: Correlation Map (Axial Slice z={mid_k})"
            elif data_shape[1] > 1:
                slice_data = corr_map[:, mid_j, :]
                title = f"{label}: Correlation Map (Coronal Slice y={mid_j})"
            else:
                slice_data = corr_map[mid_i, :, :]
                title = f"{label}: Correlation Map (Sagittal Slice x={mid_i})"

            im = ax_slice.imshow(slice_data, cmap="viridis", origin="lower", vmin=0, vmax=1)
            ax_slice.set_title(title)
            plt.colorbar(im, ax=ax_slice, fraction=0.046, pad=0.04)

            # Add summary stats as text
            overall_corr = multi_results[method]["overall_correlation"]
            overall_mae = multi_results[method]["overall_mae"]
            stats_text = (
                f"Mean corr: {mean_corr:.4f}\n"
                f"Overall corr: {overall_corr:.4f}\n"
                f"MAE: {overall_mae:.4f}"
            )
            ax_hist.text(
                0.02, 0.98,
                stats_text,
                transform=ax_hist.transAxes,
                fontsize=9,
                verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8)
            )

        plt.tight_layout()
        return fig

    plot_multi_voxel_results(multi_results, data.shape)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 7. Summary Statistics

    Compare performance between methods
    """)
    return


@app.cell
def _(multi_results, results):
    print("=== ROUND-TRIP PREDICTION SUMMARY ===\n")

    print("Single Voxel Results:")
    for _method in ["standard", "gqi2"]:
        result = results[_method]
        print(
            f"  {_method.title():8s}: Correlation = {result['correlation']:.4f}, MAE = {result['mae']:.4f}"
        )

    print("\nMulti-Voxel Results:")
    for _method in ["standard", "gqi2"]:
        result = multi_results[_method]
        print(
            f"  {_method.title():8s}: Avg Correlation = {result['avg_correlation']:.4f}, "
            f"Overall Correlation = {result['overall_correlation']:.4f}, "
            f"Overall MAE = {result['overall_mae']:.4f}"
        )


    # Determine which method performed better
    std_corr = multi_results["standard"]["avg_correlation"]
    gqi2_corr = multi_results["gqi2"]["avg_correlation"]

    if std_corr > gqi2_corr:
        print(f"\nStandard method performed better ({std_corr:.4f} vs {gqi2_corr:.4f})")
    elif gqi2_corr > std_corr:
        print(f"\nGQI2 method performed better ({gqi2_corr:.4f} vs {std_corr:.4f})")
    else:
        print(f"\nBoth methods performed similarly ({std_corr:.4f})")
    return


if __name__ == "__main__":
    app.run()
