import re
from pathlib import Path
from typing import List

import h5py
import numpy as np
import typer
from pyproj import Transformer
from openquake.hazardlib.tests.geo.line_test import get_mesh

from pygmt_helper import plotting as pygmt_plt
from gmhazard_2 import source_model
from gmhazard_2 import plotting
from gmhazard_2 import dbs
from gmhazard_2 import distance
from gmhazard_2 import utils


app = typer.Typer()


@app.command("plot-rupture-sections")
def plot_rupture_sections(
    rupture_sections_ffp: Path, output_dir: Path, plot_data_ffp: Path = None
):
    """Plots the rupture sections"""
    sections = source_model.parse_rupture_sections(rupture_sections_ffp)

    region = plotting.get_region_from_sections(
        sections, lon_bounds=0.25, lat_bounds=0.25
    )

    # Load plot data
    plot_data = None
    if plot_data_ffp is not None:
        plot_data = pygmt_plt.NZMapData.load(plot_data_ffp, high_res_topo=False)

    # Generate plot
    fig = pygmt_plt.gen_region_fig(
        region=region,
        map_data=plot_data,
    )
    plotting.plot_sections(fig, sections)
    fig.savefig(
        output_dir / f"{rupture_sections_ffp.stem}.png", dpi=900, anti_alias=True
    )


@app.command("plot-rupture-scenarios")
def plot_rupture_scenarios(
    rupture_dir: Path,
    output_dir: Path,
    plot_data_dir: Path = None,
    n_scenarios: int = 100,
    random: bool = True,
    zoomed: bool = True,
):
    """
    Creates GMT based plots for the different
    rupture scenarios
    """
    plotting.create_scenario_plots(
        rupture_dir,
        output_dir=output_dir,
        plot_data_dir=plot_data_dir,
        n_scenarios=n_scenarios,
        random=random,
        zoomed=zoomed,
    )


@app.command("gen-distances-mesh-plot")
def plot_rx_ry_mesh(source_model_db_ffp: Path, scenario_id: int, output_dir: Path):
    (
        segment_nztm_coords,
        p_x,
        p_y,
        scenario_Rx,
        scenario_Ry,
        scenario_Rrup,
        scenario_Rjb,
    ) = utils.compute_scenario_mesh_distances(source_model_db_ffp, scenario_id)

    # Rx
    fig = plotting.create_distance_plot(segment_nztm_coords, p_x, p_y, scenario_Rx, -30, 30, "coolwarm",
                                  plot_downdip_points=True, title=f"{scenario_id} - Rx")
    fig.savefig(output_dir / f"{scenario_id}_Rx.png")

    # Ry
    fig = plotting.create_distance_plot(segment_nztm_coords, p_x, p_y, scenario_Ry, -30, 30, "coolwarm",
                                  plot_downdip_points=True, title=f"{scenario_id} - Ry")
    fig.savefig(output_dir / f"{scenario_id}_Ry.png")

    # Rrup
    fig = plotting.create_distance_plot(segment_nztm_coords, p_x, p_y, scenario_Rrup, -30, 30, "coolwarm",
                                    plot_downdip_points=True, title=f"{scenario_id} - Rrup")
    fig.savefig(output_dir / f"{scenario_id}_Rrup.png")

    # Rjb
    fig = plotting.create_distance_plot(segment_nztm_coords, p_x, p_y, scenario_Rjb, -30, 30, "coolwarm",
                                    plot_downdip_points=True, title=f"{scenario_id} - Rjb")
    fig.savefig(output_dir / f"{scenario_id}_Rjb.png")


if __name__ == "__main__":
    app()
