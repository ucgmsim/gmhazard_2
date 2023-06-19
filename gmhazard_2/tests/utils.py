from typing import Dict

import numpy as np
import pandas as pd

from openquake.hazardlib.tests.geo.line_test import get_mesh
from openquake.hazardlib.geo.multiline import MultiLine
from openquake.hazardlib.geo.line import Line

from .. import utils


def process_rupture_scenario(
    rupture_scenarios_df: pd.DataFrame,
    scenario_id: str,
    segment_coords: np.ndarray,
    segment_nztm_coords: np.ndarray,
    segment_section_ids: np.ndarray,
    lines: Dict[str, Line],
):
    """Helper function to perform mesh based distance
    calculation for a given rupture"""
    cur_section_ids = rupture_scenarios_df.loc[scenario_id].section_ids
    cur_scenario_segment_mask = np.isin(segment_section_ids, cur_section_ids)

    cur_segment_coords = segment_coords[:, :, cur_scenario_segment_mask]

    # Site mesh
    mesh, plons, plats = get_mesh(
        cur_segment_coords[:, 0, :].ravel().min() - 0.1,
        cur_segment_coords[:, 0, :].ravel().max() + 0.1,
        cur_segment_coords[:, 1, :].ravel().min() - 0.1,
        cur_segment_coords[:, 1, :].ravel().max() + 0.1,
        0.01,
    )

    # Perform distance calculation
    (
        scenario_Rx,
        scenario_Ry,
        scenario_Rrup,
        scenario_Rjb,
    ) = utils.compute_mesh_distances(
        segment_nztm_coords[:, :, cur_scenario_segment_mask],
        plons,
        plats,
        cur_section_ids,
        segment_section_ids[cur_scenario_segment_mask],
        ll=True,
    )

    # Create the OQ MultiLine object
    cur_lines = [lines[cur_id] for cur_id in cur_section_ids]
    ml = MultiLine(cur_lines)
    ml.set_tu(mesh)
    uupp = ml.uut.reshape(plons.shape)
    tupp = ml.tut.reshape(plons.shape)

    return tupp, uupp, scenario_Rx, scenario_Ry, plons, plats, cur_segment_coords
