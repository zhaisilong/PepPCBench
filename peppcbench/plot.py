from typing import Optional
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


def plot_contact_map(cmap, title="Contact Map", save_path=None, cmap_name="Blues"):
    """Plots a contact map using Matplotlib with probability-based color intensity.

    Args:
        cmap (np.ndarray): A square contact map matrix (num_residues, num_residues).
        title (str): Title of the plot.
        save_path (str, optional): If provided, saves the figure to this path.
    """
    plt.figure(figsize=(8, 8))

    # Plot using color intensity (higher probability = darker color)
    plt.imshow(cmap, cmap=cmap_name, origin="upper", interpolation="nearest")

    # Add colorbar to show probability scale
    cbar = plt.colorbar()
    cbar.set_label("Contact Probability")

    plt.xlabel("Residue Index")
    plt.ylabel("Residue Index")
    plt.title(title)

    # Draw diagonal for self-contacts reference
    plt.plot(
        np.arange(cmap.shape[0]),
        np.arange(cmap.shape[0]),
        color="red",
        linestyle="dashed",
        linewidth=0.5,
    )
    plt.tight_layout()

    # Save the plot if a path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()
    plt.close()


def plot_matrix(
    actifptm_dict, iptm_dict, cptm_dict, prefix="rank", ax_in=None, fig_path=None
):
    if not ax_in:  # In case, we are not plotting multiple models next to each other
        fig, ax = plt.subplots(1, 1, figsize=(5, 5), squeeze=False)

    letters = sorted(
        set(
            [key.split("-")[0] for key in actifptm_dict.keys()]
            + [key.split("-")[1] for key in actifptm_dict.keys()]
        )
    )

    data = pd.DataFrame(
        np.zeros((len(letters), len(letters))), index=letters, columns=letters
    )

    for key, value in actifptm_dict.items():
        i, j = key.split("-")
        data.loc[j, i] = value
        data.loc[i, j] = iptm_dict[f"{i}-{j}"]

    for chain, value in cptm_dict.items():
        if chain in data.index:
            data.loc[chain, chain] = value

    mask_upper = np.triu(np.ones(data.shape), k=1)
    mask_lower = np.tril(np.ones(data.shape), k=-1)
    mask_diagonal = np.eye(data.shape[0])

    dyn_size_ch = max(
        -1.5 * len(letters) + 18, 3
    )  # resize the font with differently sized figures
    # Plot lower triangle (actifpTM)
    ax_in.imshow(
        np.ma.masked_where(mask_upper + mask_diagonal, data),
        cmap="Blues",
        vmax=1,
        vmin=0,
    )

    # Plot upper triangle (ipTM)
    ax_in.imshow(
        np.ma.masked_where(mask_lower + mask_diagonal, data),
        cmap="Reds",
        vmax=1,
        vmin=0,
    )

    # Plot diagonal (cpTM)
    diagonal_data = np.diag(np.diag(data))
    im = ax_in.imshow(
        np.ma.masked_where(~mask_diagonal.astype(bool), diagonal_data),
        cmap="Greys",
        vmax=1,
        vmin=0,
    )

    # Add colorbar for cpTM (diagonal)
    cbar = plt.colorbar(im, ax=ax_in, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=dyn_size_ch)  # Set fontsize for colorbar labels
    cbar.outline.set_edgecolor("grey")
    cbar.outline.set_linewidth(0.5)

    # Add text annotations
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            value = data.iloc[i, j]
            if not mask_upper[i, j] and not mask_diagonal[i, j]:
                text_color = "white" if value > 0.8 else "black"
                ax_in.text(
                    j,
                    i,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=dyn_size_ch * 1.2,
                )
            elif not mask_lower[i, j] and not mask_diagonal[i, j]:
                text_color = "white" if value > 0.8 else "black"
                ax_in.text(
                    j,
                    i,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=dyn_size_ch * 1.2,
                )
            elif mask_diagonal[i, j]:
                text_color = "white" if value > 0.5 else "black"
                ax_in.text(
                    j,
                    i,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=dyn_size_ch * 1.2,
                )

    # Custom colored legend (ifpTM, cpTM, ipTM)
    x_start = 0.25
    x_offset = 0.125
    dyn_size = 16
    ax_in.text(
        x_start + 0.1,
        1.05 + x_offset,
        prefix,
        fontsize=dyn_size,
        fontweight="bold",
        color="black",
        ha="center",
        transform=ax_in.transAxes,
    )
    ax_in.text(
        x_start + x_offset - 0.06,
        1.05,
        "actifpTM",
        fontsize=dyn_size,
        fontweight="bold",
        color="darkblue",
        ha="center",
        transform=ax_in.transAxes,
    )
    ax_in.text(
        x_start + 2 * x_offset,
        1.05,
        " - ",
        fontsize=dyn_size,
        fontweight="bold",
        color="black",
        ha="center",
        transform=ax_in.transAxes,
    )
    ax_in.text(
        x_start + 3 * x_offset,
        1.05,
        "cpTM",
        fontsize=dyn_size,
        fontweight="bold",
        color="dimgrey",
        ha="center",
        transform=ax_in.transAxes,
    )
    ax_in.text(
        x_start + 4 * x_offset,
        1.05,
        " - ",
        fontsize=dyn_size,
        fontweight="bold",
        color="black",
        ha="center",
        transform=ax_in.transAxes,
    )
    ax_in.text(
        x_start + 5 * x_offset,
        1.05,
        "ipTM",
        fontsize=dyn_size,
        fontweight="bold",
        color="firebrick",
        ha="center",
        transform=ax_in.transAxes,
    )

    # Format labels
    ax_in.set_yticks(np.arange(len(letters)))
    ax_in.set_yticklabels(letters, rotation=0, fontsize=dyn_size_ch * 1.5)
    ax_in.set_xticks(np.arange(len(letters)))
    ax_in.set_xticklabels(letters, fontsize=dyn_size_ch * 1.5)

    # If this was only one plot, display and save it.
    # If multiple plots have been appended, this needs to be done from the calling function
    if not ax_in:
        plt.tight_layout()
        plt.savefig(fig_path, dpi=200, bbox_inches="tight")


def plot_chain_pairwise_analysis(info, fig_path: Optional[str] = None):
    num_elements = len(info)
    max_plots_per_row = 5
    num_rows = (
        num_elements + max_plots_per_row - 1
    ) // max_plots_per_row  # Ceiling division to calculate rows needed

    # Define subplot grid dynamically based on the number of elements
    fig, axes = plt.subplots(
        num_rows,
        min(num_elements, max_plots_per_row),
        figsize=(
            max_plots_per_row * 5,
            5 * num_rows,
        ),  # Adjust width always to 5 plots wide, height by number of rows
        squeeze=False,
    )
    axes = axes.flatten()  # Flatten the axes array for easy iteration

    # Iterate over all info elements and plot
    for idx, ax in enumerate(
        axes[:num_elements]
    ):  # Only use the first 'num_elements' axes
        element = info[idx]
        prefix_plot = element.get("prefix", "") if isinstance(element, dict) else ""
        actifptm_dict = (
            element.get("pairwise_actifptm", {})
            if isinstance(element, dict)
            else element[4]
        )
        iptm_dict = (
            element.get("pairwise_iptm", {})
            if isinstance(element, dict)
            else element[3]
        )
        cptm_dict = (
            element.get("per_chain_ptm", {})
            if isinstance(element, dict)
            else element[5]
        )

        plot_matrix(
            actifptm_dict,
            iptm_dict,
            cptm_dict,
            prefix=prefix_plot,
            ax_in=ax,
            fig_path=None,  # Temporarily remove to avoid saving multiple partial figures
        )

    plt.tight_layout()
    if fig_path:
        plt.savefig(fig_path, dpi=400, bbox_inches="tight")
    plt.show()
    plt.close(fig)
