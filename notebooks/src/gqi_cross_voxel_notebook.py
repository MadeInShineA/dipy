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
def _(mo, num_test_voxels, num_train_voxels):
    mo.md(f"""
    # GQI Cross-Voxel Prediction Notebook

    This notebook demonstrates **cross-voxel prediction** in GQI reconstruction:
    - For {num_test_voxels} test voxels, fit GQI model on {num_train_voxels} other train voxels
    - Predict train voxels signal using the fitted models
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

    For each test voxel, fit GQI on the train voxels and predict the test voxels
    """)
    return


@app.cell
def _():
    # Parameters
    num_test_voxels = 10
    num_train_voxels = 10
    return num_test_voxels, num_train_voxels


@app.cell
def _(
    GeneralizedQSamplingModel,
    data,
    gtab,
    np,
    num_test_voxels,
    num_train_voxels,
):
    def _():
    

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
def _(cross_results, np, num_train_voxels, plt):
    def plot_cross_voxel_results(cross_results, num_train_voxels=10):
        methods = ["gqi2", "standard"]
        method_labels = ["GQI2", "Standard GQI"]
        colors = ["#1f77b4", "#ff7f0e"]
        correlations_list = [cross_results[m]["correlations"] for m in methods]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle(
            f"Cross-Voxel Prediction Accuracy (using {num_train_voxels} training voxels per test voxel)",
            fontsize=14,
            weight='bold'
        )

        # --- Left: Violin + box + jitter ---
        parts = ax1.violinplot(
            correlations_list, 
            positions=[1, 2],
            showmeans=False,
            showmedians=False,
            widths=0.8
        )
        for pc, color in zip(parts['bodies'], colors):
            pc.set_facecolor(color)
            pc.set_alpha(0.6)

        ax1.boxplot(
            correlations_list,
            positions=[1, 2],
            widths=0.15,
            patch_artist=True,
            boxprops=dict(facecolor='white', color='black'),
            medianprops=dict(color='black'),
            whiskerprops=dict(color='black'),
            capprops=dict(color='black'),
            flierprops=dict(marker='o', markersize=3, alpha=0.5)
        )

        for i, corr in enumerate(correlations_list, start=1):
            x = np.random.normal(i, 0.04, size=len(corr))
            ax1.scatter(x, corr, alpha=0.4, color=colors[i-1], s=10)

        ax1.set_xticks([1, 2])
        ax1.set_xticklabels(method_labels)
        ax1.set_ylabel("Pearson Correlation")
        ax1.set_ylim(-0.1, 1.0)
        ax1.grid(True, linestyle='--', alpha=0.5)
        ax1.set_title("Distribution of Prediction Correlations")

        # --- Right: Density histograms ---
        for corr, label, color in zip(correlations_list, method_labels, colors):
            ax2.hist(corr, bins=25, alpha=0.6, label=label, color=color, density=True, edgecolor='black', linewidth=0.5)
        ax2.set_xlabel("Correlation")
        ax2.set_ylabel("Density")
        ax2.legend()
        ax2.grid(True, linestyle='--', alpha=0.5)
        ax2.set_title("Correlation Density Comparison")

        # --- Add summary statistics as text boxes ---
        for idx, (corr, label, color) in enumerate(zip(correlations_list, method_labels, colors)):
            mean_val = np.mean(corr)
            min_val = np.min(corr)
            max_val = np.max(corr)
            std_val = np.std(corr)

            stats_text = (
                f"{label}:\n"
                f"Mean: {mean_val:.4f}\n"
                f"Min:  {min_val:.4f}\n"
                f"Max:  {max_val:.4f}\n"
                f"Std:  {std_val:.4f}"
            )

            # Position text on right plot (adjust x/y per method)
            ax2.text(
                0.02 if idx == 0 else 0.52,  # x: left for first, right for second
                0.95 - idx * 0.5,           # y: top for first, lower for second
                stats_text,
                transform=ax2.transAxes,
                fontsize=9,
                verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.15)
            )

        plt.tight_layout()
        return fig

    plot_cross_voxel_results(cross_results, num_train_voxels=num_train_voxels)
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


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
