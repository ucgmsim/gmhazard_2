from typing import List

import numba as nb
import numpy as np

from . import distance


def compute_segment_area(segment_nztm_coords: np.ndarray):
    """
    Computes the area of a segment

    Parameters
    ----------
    segment_nztm_coords: array of floats
        Coordinates of the segment corner points in NZTM
        where points 0 and 2 define the fault trace
        shape: [4, 3, n_segments]

    Returns
    -------
    array of floats:
        The area of each segment
    """
    return (
        0.5
        * np.linalg.norm(
            np.cross(
                segment_nztm_coords[1, :, :] - segment_nztm_coords[0, :, :],
                segment_nztm_coords[2, :, :] - segment_nztm_coords[0, :, :],
                axis=0,
            ),
            axis=0,
        )
        + 0.5
        * np.linalg.norm(
            np.cross(
                segment_nztm_coords[2, :, :] - segment_nztm_coords[0, :, :],
                segment_nztm_coords[3, :, :] - segment_nztm_coords[0, :, :],
                axis=0,
            ),
            axis=0,
        )
    ) / 1e6


def compute_segment_dip(segment_nztm_coords: np.ndarray):
    """
    Compute the average segment dip
    based on the two endpoints of the segment
    """
    # Compute the opposite
    o1 = segment_nztm_coords[1, 2, :] - segment_nztm_coords[0, 2, :]
    o2 = segment_nztm_coords[3, 2, :] - segment_nztm_coords[2, 2, :]

    # Compute the adjacent
    a1 = np.linalg.norm(
        segment_nztm_coords[0, :2, :] - segment_nztm_coords[1, :2, :], axis=0
    )
    a2 = np.linalg.norm(
        segment_nztm_coords[2, :2, :] - segment_nztm_coords[3, :2, :], axis=0
    )

    # Compute the dip
    dip1 = np.degrees(np.arctan(o1 / a1))
    dip2 = np.degrees(np.arctan(o2 / a2))

    return np.mean(np.stack((dip1, dip2), axis=0), axis=0)


def compute_segment_ztor(segment_nztm_coords: np.ndarray):
    """Computes top-edge depth for each segment (i.e. ZTor)"""
    return np.mean(segment_nztm_coords[::2, 2, :], axis=0) / 1e3


def compute_section_source_props(
    segment_area: np.ndarray,
    segment_dip: np.ndarray,
    segment_ztor: np.ndarray,
    segment_section_ids: np.ndarray,
):
    """
    Compute the section area, which is just the
    sum of the segment areas
    """
    # Sanity check
    assert np.all(np.sort(segment_section_ids) == segment_section_ids)

    # Get the indices for grouping of segments
    section_ids, counts = np.unique(segment_section_ids, return_counts=True)
    reduce_ind = np.concatenate(([0], np.cumsum(counts)[:-1]))

    # Compute area and dip
    section_area = np.add.reduceat(segment_area, reduce_ind)
    section_dip = np.add.reduceat(segment_dip * segment_area, reduce_ind) / section_area
    section_ztor = np.add.reduceat(segment_ztor * segment_area, reduce_ind) / section_area

    return section_ids, section_area, section_dip, section_ztor


@nb.njit(
    nb.types.UniTuple(nb.float64[:], 2)(
        nb.int64[::1],
        nb.float64[::1],
        nb.float64[::1],
        nb.float64[::1],
        nb.types.ListType(nb.int64[::1]),
    ),
    parallel=True,
)
def compute_scenario_source_props(
    section_ids: np.ndarray,
    section_area: np.ndarray,
    section_dip: np.ndarray,
    section_ztor: np.ndarray,
    scenario_section_ids: List[np.ndarray],
):
    """
    Compute the dip and ztor for each scenario

    Is there a better way to do this???
    """
    # Sanity check
    assert section_ids.size == np.unique(section_ids).size

    scenario_dip = np.zeros(len(scenario_section_ids))
    scenario_ztor = np.zeros(len(scenario_section_ids))
    for i in nb.prange(len(scenario_section_ids)):
        cur_section_ids = scenario_section_ids[i]

        # Needed as numba doesn't support np.isin yet
        cur_scenario_dip, cur_scenario_area = 0.0, 0.0
        cur_scenario_ztor = 0.0
        for j in range(cur_section_ids.size):
            m = section_ids == cur_section_ids[j]
            cur_scenario_dip = cur_scenario_dip + (
                section_dip[m][0] * section_area[m][0]
            )
            cur_scenario_ztor = cur_scenario_ztor + (
                section_ztor[m][0] * section_area[m][0])
            cur_scenario_area = cur_scenario_area + section_area[m][0]

        scenario_dip[i] = cur_scenario_dip / cur_scenario_area
        scenario_ztor[i] = cur_scenario_ztor / cur_scenario_area

    return scenario_dip, scenario_ztor


@nb.njit(cache=True)
def compute_scenario_strike(
    trace_points: np.ndarray,
    segment_strike_vecs: np.ndarray,
    segment_trace_length: np.ndarray,
    segment_section_ids: np.ndarray,
):
    """
    Compute nominal strike across rupture scenario
    Based on Spudich et al. (2015)
    Section: Strike Discordance and Nominal Strike

    Note: As this calculation potentially flips the section strike
    vector, the segment strike vector is not modified in place
    instead a mask of the flipped strike segments is returned

    Parameters
    ----------
    trace_points: array of floats
        The coordinates (NZTM) of the trace points
        shape: [n_trace_points, 2, n_segments]
    segment_strike_vecs
        The strike vector of each segment
        shape: [2, n_segments]
    segment_trace_length
        The length of each segment
        shape: [n_segments]

    Returns
    -------
    scenario_strike_vec: array of floats
        The strike vector of the rupture scenario
        shape: [2]
    scenario_strike: float
        The strike of the rupture scenario
    scenario_origin: array of floats
        The origin of the rupture scenario
        shape: [2]
    segment_strike_flip_mask: array of bools
        A mask of the segments that have strike flipped
        shape: [n_segments]
    """
    # Make matrix of all unique trace points
    # unique_trace_points = np.unique(
    #     trace_points.transpose((0, 2, 1)).reshape((-1, 2)), axis=0
    # )

    # Numba does not support the axis keyword for np.unique
    # However, getting the subset of unique trace points
    # merely reduces the iteration for distance matrix computation,
    # therefore will just skip it for now until numba supports it.
    unique_trace_points = np.ascontiguousarray(
        trace_points.transpose((0, 2, 1))
    ).reshape((-1, 2))

    # Compute the distance matrix
    dist_matrix = np.zeros((unique_trace_points.shape[0], unique_trace_points.shape[0]))
    for i in range(unique_trace_points.shape[0]):
        # dist_matrix[i, :] = np.linalg.norm(
        #     unique_trace_points[i] - unique_trace_points, axis=1
        # )

        # Compute distance manually since numba does not support
        # axis keyword for np.linalg.norm
        coord_diff = unique_trace_points[i] - unique_trace_points
        dist_matrix[i, :] = np.sqrt(coord_diff[:, 0] ** 2 + coord_diff[:, 1] ** 2)

    # Find the trace point combination that has the maximum separation distance
    # ix_1, ix_2 = np.unravel_index(dist_matrix.argmax(), dist_matrix.shape)

    # Numba does not support unravel_index,
    # therefore implement this manually
    flat_ix = dist_matrix.argmax()
    ix_1 = flat_ix // dist_matrix.shape[1]
    ix_2 = flat_ix % dist_matrix.shape[0]

    # Compute possible vectors
    v1 = unique_trace_points[ix_1] - unique_trace_points[ix_2]
    v1 /= np.linalg.norm(v1)
    v2 = unique_trace_points[ix_2] - unique_trace_points[ix_1]
    v2 /= np.linalg.norm(v2)

    # Choose the east pointing one and compute a_hat
    a = v1 if v1[0] > 0 else v2
    a_hat = a / np.linalg.norm(a)

    ### Compute the "strike" per section/fault trace
    ### based on the equation for e_j in Spudich et al. (2015)
    ## I.e. the vector from the origin to the end of the trace

    # Get the unique section ids, has to be in the same order as
    # the sections in segment_section_id
    # Numba doesn't support np.unique with return_index=True,
    # hence manual hack
    # _, unique_section_id_ind = np.unique(segment_section_ids, return_index=True)
    unique_section_id_ind = np.concatenate(
        (np.asarray([0]), np.flatnonzero(np.diff(segment_section_ids)) + 1)
    )

    unique_section_ids = segment_section_ids[unique_section_id_ind]
    section_strike_vecs = np.zeros((2, unique_section_ids.size))
    for i, cur_section_id in enumerate(unique_section_ids):
        m = segment_section_ids == cur_section_id
        # Compute the two possible strike vectors
        v3 = trace_points[:, :, m][0, :, 0] - trace_points[:, :, m][1, :, -1]
        v4 = trace_points[:, :, m][1, :, -1] - trace_points[:, :, m][0, :, 0]

        # Compute the average segment strike vector
        avg_segment_strike_vec = (
            segment_strike_vecs[:, m] * segment_trace_length[m]
        ).sum(axis=1)
        avg_segment_strike_vec /= segment_trace_length[m].sum()

        # Choose the correct section strike vector
        if np.dot(v3 / np.linalg.norm(v3), avg_segment_strike_vec) > np.dot(
            v4 / np.linalg.norm(v4), avg_segment_strike_vec
        ):
            section_strike_vecs[:, i] = v3
        else:
            section_strike_vecs[:, i] = v4

    # Compute e_j = strike_vec . a_hat
    # e_j = np.einsum("ij,i->j", section_strike_vecs, a_hat)
    # Numba doesn't support einsum
    e_j = np.sum(section_strike_vecs * np.expand_dims(a_hat, axis=1), axis=0)

    # Compute E
    E = np.sum(e_j)

    # Switch any strike vectors with opposite sign to E
    section_strike_flip_mask = np.sign(e_j) != np.sign(E)
    if np.any(section_strike_flip_mask):
        section_strike_vecs[:, section_strike_flip_mask] = (
            -1.0 * section_strike_vecs[:, section_strike_flip_mask]
        )

    # The segments corresponding to the flipped section strike vectors
    # segment_strike_flip_mask = np.isin(
    #     segment_section_ids,
    #     segment_section_ids[unique_section_id_ind[section_strike_flip_mask]],
    # )
    # Numba doesn't support np.isin
    segment_strike_flip_mask = np.array(
        [
            True
            if id
            in segment_section_ids[unique_section_id_ind[section_strike_flip_mask]]
            else False
            for id in segment_section_ids
        ]
    )

    # Compute nominal strike
    scenario_strike_vec = np.sum(section_strike_vecs, axis=1)
    scenario_strike = np.mod(
        np.degrees(np.arctan2(scenario_strike_vec[0], scenario_strike_vec[1])),
        360,
    )
    scenario_strike_vec /= np.linalg.norm(scenario_strike_vec)

    scenario_origin = (
        unique_trace_points[ix_2]
        if np.dot(v1, scenario_strike_vec) > 0
        else unique_trace_points[ix_1]
    )

    return (
        scenario_strike_vec,
        scenario_strike,
        scenario_origin,
        section_strike_flip_mask,
        segment_strike_flip_mask,
    )


def compute_segment_strike_ll(
    segment_coords: np.ndarray, segment_nztm_coords: np.ndarray
):
    """
    Computes strike for the given segments
    using lon/lat coordinates

    Parameters
    ----------
    segment_coords: array of floats
        The coordinates of the segment corners
        Assumes that the first and third point
        define the trace of the fault with the
        and that the second and fourth point
        are the corresponding down dip points

        shape: [4, 2, n_faults], (lon, lat)
    segment_nztm_coords: array of floats
        The coordinates of the segment corners
        in NZTM coordinate system

        shape: [4, 2, n_faults], (x, y)

    Returns
    -------
    strike: array of floats
        shape: [n_points]
    """
    strike_1 = distance.ll_bearing(segment_coords[0, :2, :], segment_coords[2, :2, :])
    strike_2 = distance.ll_bearing(segment_coords[2, :2, :], segment_coords[0, :2, :])

    # Ensure that strike is in the correct direction
    v1 = segment_nztm_coords[1, :, :] - segment_nztm_coords[0, :, :]
    v2 = segment_nztm_coords[2, :, :] - segment_nztm_coords[0, :, :]
    angle_1 = np.degrees(
        np.arccos(
            np.einsum("ij, ij -> j", v1, v2)
            / (np.linalg.norm(v1, axis=0) * np.linalg.norm(v2, axis=0))
        )
    )
    bearing_1 = distance.ll_bearing(segment_coords[0, :2, :], segment_coords[1, :2, :])

    # Ensure the correct strike is used
    strike = np.where(
        np.abs((strike_1 + angle_1) - bearing_1) < 5.0, strike_1, strike_2
    )
    return strike


def compute_segment_strike_nztm(segment_nztm_coords: np.ndarray):
    """
    Computes strike for the given segments
    using NZTM coordinates

    Parameters
    ----------
    segment_nztm_coords:
        The NZTM coordinates of the segment corners
        Assumes that the first and third point
        define the trace of the fault with the
        and that the second and fourth point
        are the corresponding down dip points

        shape: [4, 2, n_faults], (x, y)

    Returns
    -------
    strike: array of floats
        shape: [n_points]
    strike_vec: array of floats
        Unit vector for the direction of strike
        shape: [2, n_points]
    """
    # Compute the two possible strike vectors
    s1 = segment_nztm_coords[2, :2, :] - segment_nztm_coords[0, :2, :]
    s1 = s1 / np.linalg.norm(s1, axis=0)
    strike_1 = np.mod(np.degrees(np.arctan2(s1[0, :], s1[1, :])), 360)

    s2 = segment_nztm_coords[0, :2, :] - segment_nztm_coords[2, :2, :]
    s2 = s2 / np.linalg.norm(s2, axis=0)
    strike_2 = np.mod(np.degrees(np.arctan2(s2[0, :], s2[1, :])), 360)

    # Compute one of the down dip vectors
    d1 = segment_nztm_coords[1, :2, :] - segment_nztm_coords[0, :2, :]
    d1 = d1 / np.linalg.norm(d1, axis=0)
    bearing_1 = np.mod(np.degrees(np.arctan2(d1[0, :], d1[1, :])), 360)
    strike_ddip_angle_1 = np.degrees(np.arccos(np.einsum("ij,ij->j", s1, d1)))

    # Choose the correct strike vector
    strike = np.where(
        (mask := np.isclose(strike_1 + strike_ddip_angle_1, bearing_1)),
        strike_1,
        strike_2,
    )
    strike_vector = np.where(mask, s1, s2)

    return strike, strike_vector
