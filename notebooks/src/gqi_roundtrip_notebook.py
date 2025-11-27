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
    def _():
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(
            f"Voxel {voxel_coord} Signal Reconstruction: Original vs. Predicted (Standard vs. GQI2 Methods)"
        )

        for idx, method in enumerate(["standard", "gqi2"]):
            result = results[method]
            original = result["original"]
            predicted = result["predicted"]

            # Scatter plot: Original vs Predicted
            ax1 = axes[idx, 0]
            ax1.scatter(original, predicted, alpha=0.6, s=20)
            ax1.plot(
                [original.min(), original.max()],
                [original.min(), original.max()],
                "r--",
                lw=2,
            )
            ax1.set_xlabel("Original Signal")
            ax1.set_ylabel("Predicted Signal")
            ax1.set_title(
                f"{method.title()} Method: Original vs Predicted\nCorrelation: {result['correlation']:.4f}"
            )
            ax1.grid(True, alpha=0.3)

            # Signal comparison plot
            ax2 = axes[idx, 1]
            x_pos = np.arange(len(original))
            width = 0.35

            ax2.bar(
                x_pos - width / 2,
                original,
                width,
                label="Original",
                alpha=0.7,
                color="blue",
            )
            ax2.bar(
                x_pos + width / 2,
                predicted,
                width,
                label="Predicted",
                alpha=0.7,
                color="red",
            )
            ax2.set_xlabel("Gradient Index")
            ax2.set_ylabel("Signal Intensity")
            ax2.set_title(
                f"{method.title()} Method: Signal Comparison\nMAE: {result['mae']:.4f}"
            )
            ax2.legend()
            ax2.grid(True, alpha=0.3)

        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave space for suptitle
        return fig

    _()
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
        valid_voxels = 0
        total_voxels = data.shape[0] * data.shape[1] * data.shape[2]

        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                for k in range(data.shape[2]):
                    original_voxel = data[i, j, k]
                    predicted_voxel = multi_predicted[i, j, k]

                    # Skip voxels with no signal
                    if np.sum(original_voxel) == 0:
                        continue

                    valid_voxels += 1
                    correlation = np.corrcoef(original_voxel, predicted_voxel)[0, 1]
                    correlations.append(correlation)

        _correlations = np.array(correlations)

        print(f"Total voxels: {total_voxels}")
        print(f"Valid voxels (with signal): {valid_voxels}")
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
            "valid_voxels": valid_voxels,
        }
    return (multi_results,)


@app.cell
def _(mo):
    mo.md("""
    ## 5. Multi-Voxel Results Visualization
    """)
    return


@app.cell
def _(multi_results, np, plt):
    def _():
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("Voxel-wise Signal Prediction Accuracy: Standard vs. GQI2 Methods")

        for idx, method in enumerate(["standard", "gqi2"]):
            result = multi_results[method]
            correlations = result["correlations"]

            # Correlation distribution histogram
            ax1 = axes[idx, 0]
            ax1.hist(correlations, bins=30, alpha=0.7, color="blue", edgecolor="black")
            ax1.axvline(
                np.mean(correlations),
                color="red",
                linestyle="--",
                linewidth=2,
                label=f"Mean: {np.mean(correlations):.3f}",
            )
            ax1.set_xlabel("Voxel-wise Correlation")
            ax1.set_ylabel("Frequency")
            ax1.set_title(f"{method.title()} Method: Correlation Distribution")
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # Correlation vs Voxel Index
            ax2 = axes[idx, 1]
            ax2.plot(correlations, alpha=0.6, linewidth=1)
            ax2.axhline(
                np.mean(correlations),
                color="red",
                linestyle="--",
                linewidth=2,
                label=f"Mean: {np.mean(correlations):.3f}",
            )
            ax2.axhline(
                0.5, color="orange", linestyle=":", linewidth=2, label="Threshold: 0.5"
            )
            ax2.set_xlabel("Voxel Index")
            ax2.set_ylabel("Correlation")
            ax2.set_title(
                f"{method.title()} Method: Correlation per Voxel\nValid voxels: {len(correlations)}"
            )
            ax2.legend()
            ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    _()
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

    print(f"\nValid voxels tested: {multi_results['standard']['valid_voxels']}")

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
