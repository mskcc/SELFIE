import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.io
from matplotlib import cm
from matplotlib.colors import ListedColormap


def load_complex(data, key):
    """Load complex data from structured numpy array."""
    return data[key]["real"] + 1j * data[key]["imag"]


def plot_and_save(image, save_path, filename, cmap="gray", vmin=None, vmax=None):
    """Plot a single image and save to disk."""
    plt.style.use("dark_background")
    plt.imshow(np.fliplr(np.rot90(image, -1)), cmap=cmap, vmin=vmin, vmax=vmax)
    plt.axis("off")
    plt.savefig(save_path / filename, transparent=True, dpi=300)
    plt.close()


def create_roi_colormap():
    """Create a custom colormap for ROI overlays."""
    cmap = cm.Spectral
    my_cmap = cmap(np.arange(cmap.N))
    my_cmap[:-1, -1] = 0  # Set alpha to 0 for all but last
    my_cmap[-1, :] = [132 / 255, 55 / 255, 107 / 255, 1]
    return ListedColormap(my_cmap)


def main():
    # --- Configuration ---
    data_path = Path(os.environ["DATA_DIR_EXP"])
    save_path = data_path / "images"
    save_path.mkdir(exist_ok=True)
    mask_folder = Path(os.environ["MASK_DIR"])

    studies = ["Study1234"]
    slice_idx = 31
    percentage_value = 95
    mode = "test"
    mode_abbrev = "test"
    norm_factor = 34
    num_coils = 8
    num_bins = 5
    frames_per_bin = 12

    for study in studies:
        # --- Load normalization factor ---
        norm_dict_path = data_path / "value90_value_dict_new.npz"
        dict_file = np.load(norm_dict_path, allow_pickle=True)
        norm_dict = dict_file[f"{mode_abbrev}_max_value_dict"].item()
        norm_value = norm_dict[f"{study}_slice_{slice_idx}"]

        # --- Load data ---
        data_slices = data_path / mode / f"{study}_slice_{slice_idx}.npz"
        data = np.load(data_slices)

        rnufft_complex = load_complex(data, "rnufft").transpose(1, 2, 0)
        rnufft_complex /= norm_value * norm_factor

        rnufft = np.abs(rnufft_complex)

        grasp = np.abs(load_complex(data, "grasp").transpose(1, 2, 0))
        grasp /= norm_value * norm_factor

        coilwise_rnufft = np.stack(
            [data["coilwise_rnufft"]["real"], data["coilwise_rnufft"]["imag"]], axis=0
        ).transpose(3, 4, 2, 1, 0)
        coilwise_rnufft /= norm_value * norm_factor

        if study == "Study1234":
            # --- Plot NUFFT and GRASP for selected time frames ---
            for tf in [0, 1, 2, 3, 59]:
                rnufft_norm = rnufft / np.percentile(rnufft, percentage_value)
                plot_and_save(
                    rnufft_norm[:, :, tf], save_path,
                    f"rnufft_tf_{tf}.png", vmin=0, vmax=1,
                )

                grasp_norm = grasp / np.percentile(grasp, percentage_value)
                plot_and_save(
                    grasp_norm[:, :, tf], save_path,
                    f"grasp_tf_{tf}.png", vmin=0, vmax=1,
                )

            # --- Plot coilwise NUFFT (real & imaginary) ---
            for coil in range(num_coils):
                plot_and_save(
                    coilwise_rnufft[:, :, coil, 0, 0], save_path,
                    f"coilwise_rnufft_tf_0_real_coil_{coil}.png",
                )
                plot_and_save(
                    coilwise_rnufft[:, :, coil, 0, 1], save_path,
                    f"coilwise_rnufft_tf_0_imag_coil_{coil}.png",
                )

                if coil == 0:
                    for tf in [1, 59]:
                        plot_and_save(
                            coilwise_rnufft[:, :, coil, tf, 0], save_path,
                            f"coilwise_rnufft_tf_{tf}_real_coil_{coil}.png",
                        )

            # --- Plot anatomical prior ---
            ana_prior = np.zeros((256, 256, num_bins))
            for i in range(num_bins):
                start = i * frames_per_bin
                end = start + frames_per_bin
                ana_prior[:, :, i] = np.abs(np.mean(rnufft_complex[:, :, start:end], axis=2))
            ana_prior /= ana_prior.max()

            for tf in range(num_bins):
                ana_prior_norm = ana_prior / np.percentile(ana_prior, percentage_value)
                plot_and_save(
                    ana_prior_norm[:, :, tf], save_path,
                    f"ana_prior_tf_{tf}.png", vmin=0, vmax=1,
                )

        # --- Load and plot ROIs ---
        masks = scipy.io.loadmat(
            mask_folder / f"ROIs_{study}_slice_{slice_idx}.mat"
        )["masks"]
        roi_cmap = create_roi_colormap()
        time_avg_rnufft = np.abs(np.mean(rnufft_complex, axis=2))
        num_rois = 3

        for roi_number in range(num_rois):
            plt.style.use("dark_background")
            plt.imshow(np.rot90(time_avg_rnufft, -1), cmap="gray")
            plt.imshow(
                np.fliplr(masks[roi_number, :, :]) + 0.1,
                cmap=roi_cmap, alpha=0.7, vmin=0.1, vmax=1,
            )
            plt.axis("off")
            plt.savefig(
                save_path / f"Study_{study}_ROI_{roi_number + 1}_slice_{slice_idx}.png",
                transparent=True, dpi=300,
            )
            plt.close()


if __name__ == "__main__":
    main()