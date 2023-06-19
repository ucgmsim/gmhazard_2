from typing import Dict
from pathlib import Path

import h5py
import pygmt
import numpy as np
import matplotlib.pyplot as plt
from pygmt_helper import plotting as pygmt_plt

from . import source_model


def create_distance_plot(
    segment_coords: np.ndarray,
    p_x: np.ndarray,
    p_y: np.ndarray,
    z: np.ndarray,
    vmin: float,
    vmax: float,
    cmap: str,
    plot_contours: bool = True,
    plot_downdip_points: bool = False,
    title: str = None,
    fig: plt.Figure = None,
    n_contour_lines: int = 10,
    equal_aspect: bool = True,
):
    """
    Creates a colormap scatter plot with the
    given segment coordinates and site coordinates

    Note: Plots are done in lon/lat space not NZTM
    Note II: Mainly for debugging and testing

    Parameters
    ----------
    segment_coords: np.ndarray
        The segment coordinates
        shape: [4, 2, n_segments]
    p_x: np.ndarray
        The site x-coordinates
        shape: [n_rows, n_cols]
    p_y: np.ndarray
        The site y-coordinates
        shape: [n_rows, n_cols]
    z: np.ndarray
        The z values
        shape: [n_rows, n_cols]
    vmin: float
        The minimum value for the colormap
    vmax: float
        The maximum value for the colormap
    """
    if fig is None:
        fig = plt.figure(figsize=(8, 8))

    plt.scatter(p_x, p_y, c=z, cmap=cmap, s=2.0, vmax=vmax, vmin=vmin)
    # plt.colorbar(pad=0)

    if plot_contours:
        cs = plt.contour(p_x, p_y, z, n_contour_lines, colors="k")
        fig.gca().clabel(cs, inline=True, fontsize=10)

    for i in range(segment_coords.shape[-1]):
        plt.plot(segment_coords[::2, 0, i], segment_coords[::2, 1, i], c="b", lw=1.0)

        if plot_downdip_points:
            plt.scatter(segment_coords[1::2, 0, :], segment_coords[1::2, 1, :], c="g", s=10.0)

    plt.title(title)
    plt.xlabel(f"X")
    plt.ylabel(f"Y")
    plt.xlim([p_x.min(), p_x.max()])
    plt.ylim([p_y.min(), p_y.max()])
    plt.grid(linewidth=0.5, alpha=0.5, linestyle="--")

    if equal_aspect:
        plt.gca().set_aspect('equal')
    plt.tight_layout()

    return fig


def plot_segment_site(segment_nztm_coords: np.ndarray, site_nztm_coords: np.ndarray):
    """
    Plots the given segment and site in NZTM coordinates

    Note: This is for debugging purposes only

    Parameters
    ----------
    segment_nztm_coords
    site_nztm_coords
    """
    fig = plt.figure(figsize=(16, 10))

    plt.scatter(site_nztm_coords[0], site_nztm_coords[1], color="red")
    plt.plot(
        segment_nztm_coords[[0, 2], 0], segment_nztm_coords[[0, 2], 1], color="black"
    )
    plt.scatter(segment_nztm_coords[:, 0], segment_nztm_coords[:, 1], color="black")

    plt.tight_layout()
    return fig


def get_region_from_sections(
    sections: Dict[int, np.ndarray], lon_bounds: float = 0, lat_bounds: float = 0
):
    """Computes the map region given the set of rupture sections"""
    min_lon, max_lon, min_lat, max_lat = np.inf, -np.inf, np.inf, -np.inf
    for cur_section in sections.values():
        min_lon = min(min_lon, cur_section[:, 0].min())
        max_lon = max(max_lon, cur_section[:, 0].max())
        min_lat = min(min_lat, cur_section[:, 1].min())
        max_lat = max(max_lat, cur_section[:, 1].max())

    return (
        min_lon - lon_bounds,
        max_lon + lon_bounds,
        min_lat - lat_bounds,
        max_lat + lat_bounds,
    )

def create_scenario_plots(
    rupture_dir: Path,
    output_dir: Path = None,
    plot_data_dir: Path = None,
    n_scenarios: int = 100,
    random: bool = True,
    zoomed: bool = True,
):
    """
    Creates GMT based plots for the different
    rupture scenarios

    TODO: Update this to use the custom source DB

    Parameters
    ----------
    rupture_dir: Path
        Path to the rupture directory
    output_dir: Path, optional
        Path to the output directory
    plot_data_dir: Path, optional
        Path to the QCore plot data directory
    n_scenarios: int, optional
        Number of scenarios to plot
    random: bool, optional
        If True, the scenarios are randomly selected
    zoomed: bool, optional
        If True, the plot is zoomed in on the rupture

    Returns
    -------
    figs: List[pygmt.Figure]
    """
    # Find the hdf5 file
    db_ffp = list(rupture_dir.glob("*hdf5"))
    assert len(db_ffp) == 1
    db_ffp = db_ffp[0]

    source_id = db_ffp.stem.rsplit("-ruptures")[0]

    # Get the rupture scenarios
    with h5py.File(db_ffp, "r") as db:
        assert len(db.keys()) == 1

        mags = db[f"{source_id}/mag"][:]
        rupture_ind = db[f"{source_id}/rupture_idxs"][:]

    rupture_ind = np.asarray(
        [
            np.asarray(cur_ind.decode().split(" ")).astype(float)
            for cur_ind in rupture_ind
        ],
        dtype=object,
    )

    # Load the sections
    sections = source_model.parse_rupture_sections(
        rupture_dir / f"{source_id}-ruptures_sections.xml"
    )

    # Get the relevant source region
    region = get_region_from_sections(
        sections, lon_bounds=0.25, lat_bounds=0.25
    )

    # Load plot data
    plot_data = None
    if plot_data_dir is not None:
        plot_data = pygmt_plt.NZMapData.load(plot_data_dir, high_res_topo=False)

    # Get the rupture scenarios to plot
    plot_rupture_ind = (
        rupture_ind[np.random.randint(0, rupture_ind.size, n_scenarios)]
        if random
        else rupture_ind[:n_scenarios]
    )

    # Plot the scenarios
    figs = []
    for i, cur_rupture_ind in enumerate(plot_rupture_ind):
        print(f"Processing rupture scenario {i + 1}/{plot_rupture_ind.size}")

        # Get relevant rupture sections
        cur_sections = {
            cur_id: cur_value
            for cur_id, cur_value in sections.items()
            if cur_id in cur_rupture_ind
        }

        # Get the current region to plot
        cur_region = (
            get_region_from_sections(
                cur_sections, lon_bounds=0.25, lat_bounds=0.25
            )
            if zoomed
            else region
        )

        # Generate plot
        fig = pygmt_plt.gen_region_fig(
            title=f"Magnitude {mags[i]:.2f}",
            region=cur_region,
            map_data=plot_data,
        )
        plot_sections(fig, cur_sections, plot_labels=True)
        figs.append(fig)

        if output_dir is not None:
            fig.savefig(
                output_dir / f"{source_id}_scenario_{i}.png", dpi=900, anti_alias=True
            )

def plot_sections(
    fig: pygmt.Figure, sections: Dict[int, np.ndarray], plot_labels: bool = False
):
    """Plots the given rupture sections"""
    for j, (id, coords) in enumerate(sections.items()):
        print(f"\tProcessing section {j}/{len(sections)}")
        for i in range(0, coords.shape[0], 2):
            fig.plot(x=coords[[i, i + 1], 0], y=coords[[i, i + 1], 1], pen="0.2p,red")

            if plot_labels:
                fig.plot(
                    x=coords[i, 0],
                    y=coords[i, 1],
                    style="x0.05c",
                    pen="black",
                    fill="black",
                )
                fig.text(
                    x=coords[i, 0],
                    y=coords[i, 1],
                    text=str(i),
                    offset="0.1c/0.1c",
                    justify="TC" if (i / 2) % 2 == 0 else "BC",
                )

                fig.plot(
                    x=coords[i + 1, 0],
                    y=coords[i + 1, 1],
                    style="x0.05c",
                    pen="black",
                    fill="black",
                )
                fig.text(
                    x=coords[i + 1, 0],
                    y=coords[i + 1, 1],
                    text=str(i + 1),
                    offset="0.1c/0.1c",
                    justify="TC" if (i / 2) % 2 == 0 else "BC",
                )

            if i <= (coords.shape[0] - 4):
                fig.plot(
                    x=coords[[i, i + 2], 0], y=coords[[i, i + 2], 1], pen="0.2p,red"
                )
                fig.plot(
                    x=coords[[i + 1, i + 3], 0],
                    y=coords[[i + 1, i + 3], 1],
                    pen="0.2p,red",
                )
