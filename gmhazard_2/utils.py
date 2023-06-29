import json
from pathlib import Path

import pandas as pd
import numpy as np
import geojson
from turfpy.measurement import points_within_polygon


from . import dbs
from . import distance
from . import source
from qcore.geo import ll_bearing

from openquake.hazardlib import nrml, sourceconverter
from openquake.hazardlib import geo


def compute_scenario_mesh_distances(source_model_db_ffp: Path, scenario_id: int):
    """
    Computes the site-to-source distances for a
    mesh of points around the specified section ids

    Parameters
    ----------
    source_model_db_ffp: Path
        Path to the source model database
    scenario_id: int
        The scenario id

    Returns
    -------
    p_x: array of float
    p_y: array of float
        The NZTM coordinates of the mesh points
    scenario_rx: array of float
    scenario_ry: array of float
    scenario_rrup: array of float
    scenario_rjb: array of float
        The distances for each meash point
    """
    with dbs.SourceModelDB(source_model_db_ffp) as db:
        rupture_section_pts_df = db.get_flt_rupture_section_pts()
        rupture_scenarios_df = db.get_fault_rupture_scenarios()

    section_ids = rupture_scenarios_df.loc[scenario_id].section_ids

    # Get the segment coordinates
    segment_coords, segment_section_ids = distance.get_segment_coords(
        rupture_section_pts_df
    )
    segment_nztm_coords = distance.segment_to_nztm(segment_coords)

    # Get the segment coordinates for the given section ids
    cur_scenario_segment_mask = np.isin(segment_section_ids, section_ids)
    segment_coords = segment_coords[:, :, cur_scenario_segment_mask]
    segment_nztm_coords = segment_nztm_coords[:, :, cur_scenario_segment_mask]
    segment_section_ids = segment_section_ids[cur_scenario_segment_mask]

    # Generate the site mesh
    # Create the mesh
    x_min, x_max = (
        segment_nztm_coords[:, 0, :].ravel().min() - 3e4,
        segment_nztm_coords[:, 0, :].ravel().max() + 3e4,
    )
    y_min, y_max = (
        segment_nztm_coords[:, 1, :].ravel().min() - 3e4,
        segment_nztm_coords[:, 1, :].ravel().max() + 3e4,
    )
    p_x, p_y = np.meshgrid(np.linspace(x_min, x_max, 50), np.linspace(y_min, y_max, 50))

    scenario_Rx, scenario_Ry, scenario_Rrup, scenario_Rjb = compute_mesh_distances(
        segment_nztm_coords, p_x, p_y, section_ids, segment_section_ids
    )

    return (
        segment_nztm_coords,
        p_x,
        p_y,
        scenario_Rx,
        scenario_Ry,
        scenario_Rrup,
        scenario_Rjb,
    )


def compute_mesh_distances(
    segment_nztm_coords: np.ndarray,
    p_x: np.ndarray,
    p_y: np.ndarray,
    section_ids: np.ndarray,
    segment_section_ids: np.ndarray,
    ll: bool = False,
):
    # Compute segment-properties
    segment_trace_length = (
        np.linalg.norm(
            segment_nztm_coords[0, :2, :] - segment_nztm_coords[2, :2, :], axis=0
        )
        / 1e3
    )
    segment_strike, segment_strike_vec = source.compute_segment_strike_nztm(
        segment_nztm_coords
    )

    # Create the result arrays
    scenario_Rx = np.full(p_x.shape, fill_value=np.nan)
    scenario_Ry = np.full(p_x.shape, fill_value=np.nan)
    scenario_Rrup = np.full(p_x.shape, fill_value=np.nan)
    scenario_Rjb = np.full(p_x.shape, fill_value=np.nan)

    segment_ry_values = np.zeros(
        (p_x.shape[0], p_x.shape[1], segment_nztm_coords.shape[2])
    )
    segment_rx_values = np.zeros(
        (p_x.shape[0], p_x.shape[1], segment_nztm_coords.shape[2])
    )

    # Compute distances for each site
    for i in range(p_x.shape[0]):
        # print(f"Processing row {i}/{p_x.shape[0]}")
        for j in range(p_x.shape[1]):
            # Get the current site coordinates
            if ll:
                site_coords = np.array([p_x[i, j], p_y[i, j]])
                site_nztm = distance.site_to_nztm(site_coords)
            else:
                site_nztm = np.array([p_x[i, j], p_y[i, j], 0])

            # Compute distances for each segment
            (
                segment_rjb,
                segment_rrup,
                segment_rx,
                segment_ry,
                segment_ry_origin,
            ) = distance.compute_segment_distances(
                segment_nztm_coords,
                segment_strike,
                segment_strike_vec,
                site_nztm,
            )

            segment_ry_values[i, j, :] = segment_ry
            segment_rx_values[i, j, :] = segment_rx

            # Get scenario Rrup and Rjb
            scenario_Rrup[i, j] = segment_rrup.min()
            scenario_Rjb[i, j] = segment_rjb.min()

            # Compute Rx and Ry for each rupture scenario
            cur_scenario_segment_mask = np.isin(segment_section_ids, section_ids)
            cur_rjb, cur_rrup, cur_T, cur_U = distance.compute_single_scenario_distances(
                section_ids,
                segment_nztm_coords,
                segment_strike_vec,
                segment_trace_length,
                segment_section_ids,
                segment_rjb,
                segment_rrup,
                segment_rx,
                segment_ry,
                segment_ry_origin,
            )

            scenario_Rx[i, j] = cur_T
            scenario_Ry[i, j] = cur_U

    return scenario_Rx, scenario_Ry, scenario_Rrup, scenario_Rjb


def create_OQ_lines(
    section_ids: np.ndarray,
    rupture_section_pts_df: pd.DataFrame,
    rupture_sections_ffp: Path,
    set_id: int,
):
    # Load the source file via OQ
    cv = sourceconverter.SourceConverter(rupture_mesh_spacing=4)
    sm = list(nrml.read_source_models([str(rupture_sections_ffp)], cv))[0]

    # Create a nshm section id -> section id lookup
    _, unique_ind = np.unique(rupture_section_pts_df.nshm_section_id, return_index=True)
    section_id_lookup = pd.Series(
        index=rupture_section_pts_df.nshm_section_id.iloc[unique_ind].values,
        data=rupture_section_pts_df.section_id.iloc[unique_ind].values,
    )

    # Get the OQ strike for each section
    oq_strike = {
        section_id_lookup[k]: cur_section.get_strike()
        for k, cur_section in sm.sections.items()
    }

    # Create the line objects for each section
    lines = {}
    for cur_section_id in section_ids:
        trace_points = (
            rupture_section_pts_df.loc[
                rupture_section_pts_df.section_id == cur_section_id
            ]
            .iloc[::2]
            .loc[:, ["lon", "lat"]]
            .values
        ).T

        # Get the strike and flip trace points if necessary
        bearing = ll_bearing(
            trace_points[0, 0],
            trace_points[1, 0],
            trace_points[0, -1],
            trace_points[1, -1],
        )
        diff = (
            d if (d := np.abs(bearing - oq_strike[cur_section_id])) < 180 else 360 - d
        )
        if diff > 90:
            trace_points = trace_points[:, ::-1]

        # Create the line object
        lines[cur_section_id] = geo.Line.from_vectors(
            trace_points[0, :], trace_points[1, :]
        )

    return lines

def get_backarc_mask(backarc_json_ffp: Path, locs: np.ndarray):
    """
    Computes a mask identifying each location
    that requires the backarc flag based on
    wether it is inside the backarc polygon or not

    locs: array of floats
        [lon, lat]
    """
    # Determine if backarc needs to be enabled for each loc
    points = geojson.FeatureCollection(
        [
            geojson.Feature(geometry=geojson.Point(tuple(cur_loc[::-1]), id=ix))
            for ix, cur_loc in enumerate(locs)
        ]
    )
    with backarc_json_ffp.open("r") as f:
        poly_coords = np.flip(json.load(f)["geometry"]["coordinates"][0], axis=1)

    polygon = geojson.Polygon([poly_coords.tolist()])
    backarc_ind = (
        [
            cur_point["geometry"]["id"]
            for cur_point in points_within_polygon(points, polygon)["features"]
        ],
    )
    backarc_mask = np.zeros(shape=locs.shape[0], dtype=bool)
    backarc_mask[backarc_ind] = True

    return backarc_mask

