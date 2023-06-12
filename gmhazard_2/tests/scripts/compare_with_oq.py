from pathlib import Path
from typing import List

import numpy as np

from gmhazard_2 import dbs
from gmhazard_2 import distance
from gmhazard_2 import plotting
from gmhazard_2 import utils
from gmhazard_2 import test_utils

import matplotlib.pyplot as plt
import typer

app = typer.Typer()


@app.command("gen-rx-ry-OQ-comparison-plots")
def gen_rx_ry_OQ_comparison_plots(
    rupture_db_ffp: Path,
    source_definitions_dir: Path,
    output_dir: Path,
    n_scenarios: int = None,
    scenario_ids: List[int] = None,
    id: str = None,
):
    if n_scenarios is None and scenario_ids is None:
        raise ValueError("Either n_scenarios or scenario_ids must be specified")

    # Load data
    with dbs.SourceModelDB(rupture_db_ffp) as db:
        source_set_info = db.get_source_set_info()
        rupture_section_pts_df = db.get_flt_rupture_section_pts()
        rupture_scenarios_df = db.get_fault_rupture_scenarios()

    # Sort out the ids..
    id = id if id is not None else source_set_info.index[0]
    id_2 = source_set_info.loc[id].id2
    set_id = source_set_info.loc[id].set_id
    rupture_sections_ffp = source_definitions_dir / id / f"{id_2}-ruptures_sections.xml"

    # Get the segment coords
    segment_coords, segment_section_ids = distance.get_segment_coords(
        rupture_section_pts_df
    )
    segment_nztm_coords = distance.segment_to_nztm(segment_coords)

    # Only interested in this set of sources
    rupture_section_pts_df = rupture_section_pts_df.loc[
        rupture_section_pts_df.set_id == set_id
    ]
    rupture_scenarios_df = rupture_scenarios_df.loc[
        rupture_scenarios_df.set_id == set_id
    ]
    section_ids = rupture_section_pts_df.section_id.unique()

    # Create the line objects for each section
    lines = utils.create_OQ_lines(
        section_ids, rupture_section_pts_df, rupture_sections_ffp, set_id
    )

    # Process each scenario
    scenario_ids = (
        scenario_ids
        if len(scenario_ids) > 0
        else rupture_scenarios_df.index.values[:n_scenarios]
    )
    for cur_scenario_id in scenario_ids:
        tupp, uupp, scenario_Rx, scenario_Ry, plons, plats, cur_segment_coords = test_utils.process_rupture_scenario(
            rupture_scenarios_df,
            cur_scenario_id,
            segment_coords,
            segment_nztm_coords,
            segment_section_ids,
            lines,
        )

        # Create current output dir
        (cur_output_dir := output_dir / str(cur_scenario_id)).mkdir(exist_ok=True)

        # Write the results
        np.save(str(cur_output_dir / "Rx_GMHazard.npy"), scenario_Rx)
        np.save(str(cur_output_dir / "Ry_GMHazard.npy"), scenario_Ry)
        np.save(str(cur_output_dir / "Rx_OQ.npy"), tupp)
        np.save(str(cur_output_dir / "Ry_OQ.npy"), uupp)

        # Plot the results
        # T
        max_abs_T = (
            np.floor(max(np.abs(scenario_Rx.min()), scenario_Rx.max()) / 10) * 10
        )
        plotting.create_distance_plot(
            cur_segment_coords,
            plons,
            plats,
            scenario_Rx,
            -max_abs_T,
            max_abs_T,
            "coolwarm",
            title="T: GMHazard",
            plot_downdip_points=True,
            equal_aspect=False,
        )
        plt.savefig(cur_output_dir / "T_GMHazard.png")
        plt.close()

        plotting.create_distance_plot(
            cur_segment_coords,
            plons,
            plats,
            tupp,
            -max_abs_T,
            max_abs_T,
            "coolwarm",
            plot_downdip_points=True,
            title="T: OQ",
            equal_aspect=False,
        )
        plt.savefig(cur_output_dir / "T_OQ.png")
        plt.close()

        # U
        max_abs_U = (
            np.floor(max(np.abs(scenario_Ry.min()), scenario_Ry.max()) / 10) * 10
        )
        plotting.create_distance_plot(
            cur_segment_coords,
            plons,
            plats,
            scenario_Ry,
            -max_abs_U,
            max_abs_U,
            "coolwarm",
            title="U: GMHazard",
            plot_downdip_points=True,
            equal_aspect=False,
        )
        plt.savefig(cur_output_dir / "U_GMHazard.png")
        plt.close()

        plotting.create_distance_plot(
            cur_segment_coords,
            plons,
            plats,
            uupp,
            -max_abs_U,
            max_abs_U,
            "coolwarm",
            plot_downdip_points=True,
            title="U: OQ",
            equal_aspect=False,
        )
        plt.savefig(cur_output_dir / "U_OQ.png")
        plt.close()

        # Plot the relative residuals
        rx_residual = np.abs(scenario_Rx - tupp)
        ry_residual = np.abs(scenario_Ry - uupp)

        plotting.create_distance_plot(
            cur_segment_coords,
            plons,
            plats,
            rx_residual,
            -1.0,
            1.0,
            "coolwarm",
            title="Rx: |GMHazard - OQ|",
            plot_downdip_points=True,
            plot_contours=False,
            equal_aspect=False,
        )
        plt.savefig(cur_output_dir / "Rx_residual.png")
        plt.close()

        plotting.create_distance_plot(
            cur_segment_coords,
            plons,
            plats,
            ry_residual,
            -1.0,
            1.0,
            "coolwarm",
            title="Ry: |GMHazard - OQ|",
            plot_downdip_points=True,
            plot_contours=False,
            equal_aspect=False,
        )
        plt.savefig(cur_output_dir / "Ry_residual.png")
        plt.close()


if __name__ == "__main__":
    app()
