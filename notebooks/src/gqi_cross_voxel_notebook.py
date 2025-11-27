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
    # GQI Cross-Voxel Prediction Notebook

    This notebook demonstrates **cross-voxel prediction** in GQI reconstruction:
    - For each voxel, fit GQI model on all other voxels
    - Predict the excluded voxel's signal using the fitted models
    - Compare predicted signals to the actual excluded voxel's signal

    **Key Concepts:**
    - Cross-voxel prediction tests if ODFs from other voxels can predict signals in a given voxel
    - High correlation indicates similar fiber orientations across neighboring voxels
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
    ## 2. Cross-Voxel Test

    For each voxel, fit GQI on all other voxels and predict the excluded voxel
    """)
    return


@app.cell
def _(GeneralizedQSamplingModel, data, gtab, np):
    def _():
        # Parameters
        num_test_voxels = 10
        num_train_voxels = 10

        # Test both methods
        cross_results = {}

        for method in ["standard", "gqi2"]:
            print(f"\n=== Cross-voxel {method} method ===")

            # Collect all possible test voxels with signal
            all_test_voxels = []
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    for k in range(data.shape[2]):
                        voxel = (i, j, k)
                        if np.sum(data[voxel]) > 0:
                            all_test_voxels.append(voxel)

            # Randomly select up to num_test_voxels
            if len(all_test_voxels) > num_test_voxels:
                selected_test_voxels = np.random.choice(
                    len(all_test_voxels), num_test_voxels, replace=False
                )
                test_voxels = [all_test_voxels[idx] for idx in selected_test_voxels]
            else:
                test_voxels = all_test_voxels

            # For each selected test voxel, fit on selected train voxels and predict the test voxel
            correlations = []

            for test_voxel in test_voxels:
                test_data = data[test_voxel]

                # Collect all possible train voxels (not test and have signal)
                train_candidates = []
                for i in range(data.shape[0]):
                    for j in range(data.shape[1]):
                        for k in range(data.shape[2]):
                            voxel = (i, j, k)
                            if voxel != test_voxel and np.sum(data[voxel]) > 0:
                                train_candidates.append(voxel)

                # Randomly select up to num_train_voxels
                if len(train_candidates) > num_train_voxels:
                    selected_train = np.random.choice(
                        len(train_candidates), num_train_voxels, replace=False
                    )
                    train_voxels = [train_candidates[idx] for idx in selected_train]
                else:
                    train_voxels = train_candidates

                # Collect correlations from selected train voxels
                voxel_correlations = []

                for train_voxel in train_voxels:
                    train_data = data[train_voxel]

                    # Fit on train_voxel
                    gq = GeneralizedQSamplingModel(
                        gtab, method=method, sampling_length=1.2
                    )
                    fit = gq.fit(train_data)

                    # Predict signal
                    predicted_signal = fit.predict(gtab)

                    # Correlate with test_data
                    correlation = np.corrcoef(predicted_signal, test_data)[0, 1]
                    voxel_correlations.append(correlation)

                if voxel_correlations:
                    avg_corr = np.mean(voxel_correlations)
                    print(
                        f"  Test voxel {test_voxel}: {len(voxel_correlations)} training voxels, avg correlation {avg_corr:.4f}"
                    )
                    correlations.append(avg_corr)

            correlations = np.array(correlations)

            print(f"Number of test voxels: {len(correlations)}")
            print(f"Average correlation: {np.mean(correlations):.4f}")
            print(f"Min correlation: {np.min(correlations):.4f}")
            print(f"Max correlation: {np.max(correlations):.4f}")
            print(f"Std correlation: {np.std(correlations):.4f}")

            cross_results[method] = {
                "correlations": correlations,
            }
        return cross_results

    cross_results = _()
    return (cross_results,)


@app.cell
def _(mo):
    mo.md("""
    ## 3. Cross-Voxel Visualization
    """)
    return


@app.cell
def _(cross_results, np, plt):
    def _():
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(
            "Cross-Voxel Prediction Accuracy per Voxel (using 10 training voxels): Standard vs. GQI2 Methods"
        )

        for idx, method in enumerate(["standard", "gqi2"]):
            result = cross_results[method]
            correlations = result["correlations"]

            # Correlation distribution histogram
            ax1 = axes[idx, 0]
            ax1.hist(correlations, bins=30, alpha=0.7, color="green", edgecolor="black")
            ax1.axvline(
                np.mean(correlations),
                color="red",
                linestyle="--",
                linewidth=2,
                label=f"Mean: {np.mean(correlations):.3f}",
            )
            ax1.set_xlabel("Cross-Voxel Correlation")
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
    ## 4. Summary Statistics

    Compare performance between methods
    """)
    return


@app.cell
def _(cross_results, np):
    print("=== CROSS-VOXEL PREDICTION SUMMARY ===\n")

    print("Cross-Voxel Results:")
    for _method in ["standard", "gqi2"]:
        result = cross_results[_method]
        avg_corr = np.mean(result["correlations"])
        print(f"  {_method.title():8s}: Average Correlation = {avg_corr:.4f}")

    # Determine which method performed better
    std_corr = np.mean(cross_results["standard"]["correlations"])
    gqi2_corr = np.mean(cross_results["gqi2"]["correlations"])

    if std_corr > gqi2_corr:
        print(f"\nStandard method performed better ({std_corr:.4f} vs {gqi2_corr:.4f})")
    elif gqi2_corr > std_corr:
        print(f"\nGQI2 method performed better ({gqi2_corr:.4f} vs {std_corr:.4f})")
    else:
        print(f"\nBoth methods performed similarly ({std_corr:.4f})")
    return


if __name__ == "__main__":
    app.run()
