import time
from typing import Tuple
import numba as nb
import numpy as np
import pandas as pd

from pyproj import Transformer
from . import plotting


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


@nb.njit
def check_site_in_segment(
    segment_nztm_coords: np.ndarray,
    site_nztm_coords: np.ndarray,
):
    """
    Checks if a point is in a segment
    defined by 4 coordinates

    Computes the area of the 4 triangles formed
    by the corner points and P
    and if the sum of the areas is equal to
    the area of the segment then
    the point is in the segment

    Parameters
    ----------
    segment_nztm_coords: array of floats
        Corner points of the segment
        shape: [4, 3], (x, y, 0)
    site_nztm_coords: array of floats
        Coordinates of points of interest
        (x, y, 0)

    Returns
    -------
    bool:
        True if point is in segment
    """
    # Compute total area of P-triangles
    index_combs = [(0, 1), (1, 3), (3, 2), (2, 0)]
    p_total_area = (
        np.sum(
            np.asarray(
                [
                    0.5
                    * np.linalg.norm(
                        np.cross(
                            segment_nztm_coords[i] - site_nztm_coords,
                            segment_nztm_coords[j] - site_nztm_coords,
                        )
                    )
                    for i, j in index_combs
                ]
            )
        )
        / 1e6
    )

    # Compute area of segment if needed
    segment_area = (
        0.5
        * np.linalg.norm(
            np.cross(
                segment_nztm_coords[1] - segment_nztm_coords[0],
                segment_nztm_coords[3] - segment_nztm_coords[0],
            )
        )
        + 0.5
        * np.linalg.norm(
            np.cross(
                segment_nztm_coords[2] - segment_nztm_coords[0],
                segment_nztm_coords[3] - segment_nztm_coords[0],
            )
        )
    ) / 1e6

    # Since math.isclose and numpy.isclose are currently not supported by numba
    return np.abs(p_total_area - segment_area) < 1e-6


@nb.njit
def f(A: np.ndarray, B: np.ndarray, t: float):
    return (1 - t) * A + t * B


@nb.njit
def g(u: np.ndarray, v: np.ndarray, t: float):
    return t ** 2 * np.dot(v, v) + 2 * t * np.dot(u, v) + np.dot(u, u)


@nb.njit(
    nb.float64(nb.float64[:, :], nb.float64[:], nb.int64[:], nb.int64[:]),
    cache=True,
)
def compute_min_line_segment_distance(
    segment_coords: np.ndarray,
    site_coords: np.ndarray,
    A_ind: np.ndarray,
    B_ind: np.ndarray,
):
    """
    Computes minimum distance to each line segment
    and then returns the minimum of those distances

    # Based on https://math.stackexchange.com/a/2193733/1180135

    Parameters
    ----------
    segment_coords: array of floats
        The coordinates of the segment corners
    site_coords: array of floats
        The coordinates of the site in NZTM
    A_ind: array of ints
        The indices of the A points in segment_coords
        I.e. Specify the start of each line segment
    B_ind
        The indices of the B points in segment_coords
        I.e. Specify the end of each line segment

    Returns
    -------
    float
    """
    A = segment_coords[A_ind, :]
    B = segment_coords[B_ind, :]

    v = B - A
    u = A - site_coords

    D = np.zeros(4)
    for i in range(4):
        t = -np.dot(u[i], v[i]) / np.dot(v[i], v[i])

        if 0 < t < 1:
            C = f(A[i], B[i], t)
        else:
            C = A[i] if g(u[i], v[i], 0) < g(u[i], v[i], 1) else B[i]

        D[i] = np.linalg.norm(C - site_coords) / 1e3

    return np.min(D)


def ll_bearing(point_1: np.ndarray, point_2: np.ndarray):
    # Calculate the bearing
    lat1 = np.radians(point_1[1, :])
    lat2 = np.radians(point_2[1, :])
    dlon = np.radians(point_2[0, :] - point_1[0, :])

    y = np.sin(dlon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)

    # Convert to 0 - 360
    return np.mod(np.degrees(np.arctan2(y, x)), 360)


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
    strike_1 = ll_bearing(segment_coords[0, :2, :], segment_coords[2, :2, :])
    strike_2 = ll_bearing(segment_coords[2, :2, :], segment_coords[0, :2, :])

    # Ensure that strike is in the correct direction
    v1 = segment_nztm_coords[1, :, :] - segment_nztm_coords[0, :, :]
    v2 = segment_nztm_coords[2, :, :] - segment_nztm_coords[0, :, :]
    angle_1 = np.degrees(
        np.arccos(
            np.einsum("ij, ij -> j", v1, v2)
            / (np.linalg.norm(v1, axis=0) * np.linalg.norm(v2, axis=0))
        )
    )
    bearing_1 = ll_bearing(segment_coords[0, :2, :], segment_coords[1, :2, :])

    # Ensure the correct strike is used
    strike = np.where(
        np.abs((strike_1 + angle_1) - bearing_1) < 5.0, strike_1, strike_2
    )
    return strike


def compute_min_line_distance(line_points: np.ndarray, site_coords: np.ndarray):
    """
    Computes the distance between a line and a point in 2D

    Parameters
    ----------
    line_points: array of floats
        The coordinates of two points on the line
        shape: [2, 2, :], (x, y, num_lines)
    site_coords:
        The coordinates of the site
        shape: [2], (x, y)

    Returns
    -------
    array of floats
        Distance to each line
    """
    numerator = np.abs(
        (line_points[1, 0, :] - line_points[0, 0, :])
        * (line_points[1, 1, :] - line_points[0, 1, :])
        - (line_points[0, 0, :] - site_coords[0])
        * (line_points[1, 1, :] - site_coords[1])
    )
    denominator = np.sqrt(
        (line_points[1, 0, :] - line_points[0, 0, :]) ** 2
        + (line_points[1, 1, :] - line_points[0, 1, :]) ** 2
    )
    return numerator / denominator


def compute_closest_point_on_line(
    line_point_1: np.ndarray, line_point_2: np.ndarray, site: np.ndarray
):
    """
    Computes the closest point on a line to a given point

    Parameters
    ----------
    line_point_1: array of floats
    line_point_2: array of floats
        The two points defining the line
        shape: [2, n_lines], (x, y)
    site: array of floats
        The site to compute the closest point for
    Returns
    -------
    array of floats
        Closest point for each line
    """
    line_vector = line_point_2 - line_point_1
    lp1_site_vector = site[:, np.newaxis] - line_point_1

    # Compute the projection of the line point - site vector onto the line vector
    line_projection = np.einsum(
        "ij, ij -> j", line_vector, lp1_site_vector
    ) / np.einsum("ij, ij -> j", line_vector, line_vector)

    # Compute the closest point
    return line_point_1 + line_projection * line_vector


@nb.njit(
    nb.types.UniTuple(nb.float64[:], 2)(nb.float64[:, :, :], nb.float64[:]),
    cache=True,
)
def compute_segment_rjb_rrup(segment_nztm_coords: np.ndarray, site_nztm: np.ndarray):
    rrup_values = np.zeros(segment_nztm_coords.shape[-1])
    rjb_values = np.zeros(segment_nztm_coords.shape[-1])

    # Create a surface projection of segments
    surface_segment_nztm_coords = segment_nztm_coords.copy()
    surface_segment_nztm_coords[:, 2, :] = 0.0

    for i in range(segment_nztm_coords.shape[-1]):
        cur_segment_nztm_coords = segment_nztm_coords[:, :, i]

        # Check if points is on the segment
        if check_site_in_segment(surface_segment_nztm_coords[:, :, i], site_nztm):
            # Compute distance to the plane
            v1 = cur_segment_nztm_coords[1, :] - cur_segment_nztm_coords[0, :]
            v2 = cur_segment_nztm_coords[2, :] - cur_segment_nztm_coords[0, :]
            n = np.cross(v1, v2)
            rrup_values[i] = (
                np.dot(n, (site_nztm - cur_segment_nztm_coords[0, :]))
                / np.linalg.norm(n)
            ) / 1e3
            rjb_values[i] = 0

        # Compute distances
        A_ind = np.asarray([0, 0, 3, 3])
        B_ind = np.asarray([1, 2, 1, 2])
        rrup_values[i] = compute_min_line_segment_distance(
            cur_segment_nztm_coords, site_nztm, A_ind, B_ind
        )
        rjb_values[i] = compute_min_line_segment_distance(
            cur_segment_nztm_coords[:, :2], site_nztm[:2], A_ind, B_ind
        )

    return rjb_values, rrup_values


def get_segment_coords(rupture_section_pts_df: pd.DataFrame):
    """
    Gets the 4 coordinates for each rupture
    section segment

    Parameters
    ----------
    rupture_section_pts_df: dataframe
        Contains the rupture section points
        One row per point, assumes these are in order
        Required columns: [lon, lat, depth, section_id]

    Returns
    -------
    segment_coords: array of floats
        shape: [4, 3, n_segments]
        where the order of points (i.e. axis=0) is
        the same as in the input dataframe and
        columns (i.e. axis=1) are in order
        [lon, lat, depth]
    segment_section_ids: array of ints
        shape: [n_segments]
        Section ID for each segment
    """
    unique_section_ids, counts = np.unique(
        rupture_section_pts_df.section_id, return_counts=True
    )
    n_segments = int(np.sum(((counts - 4) / 2) + 1))

    # Get the relevant values
    section_ids = rupture_section_pts_df.section_id.values
    section_coords = rupture_section_pts_df[["lon", "lat", "depth"]].values

    # Get segment coordinates
    ix = 0
    segment_coords = np.zeros((4, 3, n_segments))
    segment_section_ids = np.zeros(n_segments, dtype=int)
    for i in range(3, rupture_section_pts_df.shape[0], 2):
        if section_ids[i - 3] != section_ids[i]:
            continue
        segment_coords[:, :, ix] = section_coords[i - 3 : i + 1]
        segment_section_ids[ix] = section_ids[i]
        ix += 1

    return segment_coords, segment_section_ids


def site_to_nztm(site_coords: np.ndarray):
    """
    Converts the site coordinates
    from lon, lat to NZTM, i.e. x, y
    and adds depth dimension (assumed to be zero)

    Parameters
    ----------
    site_coords: array of floats
        shape: [2]

    Returns
    -------
    site_nztm_coords: array of floats
        shape: [3]
    """
    transformer = Transformer.from_crs(4326, 2193, always_xy=True)
    x, y = transformer.transform(site_coords[0], site_coords[1])
    return np.asarray([x, y, 0])


def segment_to_nztm(segment_coords: np.ndarray):
    """
    Converts the segment corner coordinates
    from lon, lat, depth to NZTM, i.e. x, y, depth

    Parameters
    ----------
    segment_coords: array of floats
        shape: [4, 3, n_segments]

    Returns
    -------
    segment_nztm_coords: array of floats
        shape: [4, 3, n_segments]
    """
    # Convert to NZTM coordinate system
    transformer = Transformer.from_crs(4326, 2193, always_xy=True)
    segment_nztm_coords = segment_coords.copy()
    a, b = transformer.transform(
        segment_coords[:, 0, :].ravel(), segment_coords[:, 1, :].ravel()
    )
    segment_nztm_coords[:, 0, :] = a.reshape(segment_coords[:, 0, :].shape)
    segment_nztm_coords[:, 1, :] = b.reshape(segment_coords[:, 1, :].shape)
    segment_nztm_coords[:, 2, :] = segment_coords[:, 2, :] * 1e3

    return segment_nztm_coords


def to_ll_coords(nztm_coords: np.ndarray):
    """
    Converts nztm coordinates to lon, lat

    Parameters
    ----------
    nztm_coords: array of floats
        shape: [2, n_points]

    Returns
    -------
    coords: array of floats
        shape: [2, n_points]
    """
    transformer = Transformer.from_crs(2193, 4326, always_xy=True)
    return np.stack(transformer.transform(nztm_coords[0, :], nztm_coords[1, :]), axis=0)


def compute_segment_rx_ry(
    segment_nztm_coords: np.ndarray,
    strike: np.ndarray,
    segment_strike_vec: np.ndarray,
    site_nztm_coords: np.ndarray,
):
    """
    Computes Rx and Ry for each segment

    Parameters
    ----------
    segment_nztm_coords: array of floats
        shape: [4, 3, n_segments]
    strike: array of floats
        shape: [n_segments]
    segment_strike_vec: array of floats
        shape: [2, n_segments]
    site_nztm_coords: array of floats
        shape: [3]

    Returns
    -------
    rx_values: array of floats
        shape: [n_segments]
    ry_values: array of floats
        shape: [n_segments]
    ry_origins: array of floats
        shape: [2, n_segments]
        Origin used for each Ry calculation
    """
    # Compute the closest point on (extended) fault trace
    # with respect to the site and convert to lon, lat
    closest_point_nztm = compute_closest_point_on_line(
        segment_nztm_coords[0, :2, :],
        segment_nztm_coords[2, :2, :],
        site_nztm_coords[:2],
    )

    ###
    ### Compute Rx
    ###
    # Compute the Rx distance (without sign)
    rx_values = (
        np.linalg.norm(closest_point_nztm - site_nztm_coords[:2, np.newaxis], axis=0)
        / 1e3
    )

    # Compute the bearing of the closest point to the site
    v1 = site_nztm_coords[:2, np.newaxis] - closest_point_nztm
    bearing = np.mod(np.degrees(np.arctan2(v1[0], v1[1])), 360)

    # If strike + 90 == bearing then rx is +ve otherwise -ve
    rx_values = np.where(
        np.abs(np.mod(strike + 90, 360) - bearing) < 5, rx_values, -rx_values
    )

    ###
    ### Compute Ry
    ###
    # Find the origin for each segment
    v2 = segment_nztm_coords[2, :2, :] - segment_nztm_coords[0, :2, :]
    # Dot product is positive if vectors point in the same
    # direction, negative if they point in opposite directions
    mask = np.einsum("ij,ij->j", v2, segment_strike_vec) > 0
    ry_origins = np.where(
        mask, segment_nztm_coords[0, :2, :], segment_nztm_coords[2, :2, :]
    )

    # Compute the magnitude of Ry
    ry_values = np.linalg.norm(ry_origins - closest_point_nztm, axis=0) / 1e3

    # Compute the sign of Ry
    origin_site_vec = site_nztm_coords[:2, np.newaxis] - ry_origins
    ry_sign = np.where(
        np.einsum("ij,ij->j", origin_site_vec, segment_strike_vec) > 0, 1, -1
    )

    return rx_values, ry_sign * ry_values, ry_origins


def compute_segment_distances(
    segment_nztm_coords: np.ndarray,
    segment_strike: np.ndarray,
    segment_strike_vec: np.ndarray,
    site_nztm_coords: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes Rrup, Rjb to each rupture section segment
    from the site

    Parameters
    ----------
    segment_nztm_coords: array of floats
        The coordinates (NZTM) for the 4 points defining each segment
        shape: [4, 3, n_segments]
    segment_strike: array of floats
        The strike of each segment
        shape: [n_segments]
    segment_strike_vec: array of floats
        The strike vector of each segment
        shape: [2, n_segments]
    segment_section_ids: array of ints
        Section ID for each segment
        shape: [n_segments]
    site_ll_coords: array of floats
        The coordinates (lon, lat) of the site
        shape: [2]
    site_nztm_coords: array of floats
        The coordinates (NZTM) of the site
        shape: [2]

    Returns
    -------
    rjb_values: array of floats
        shape: [n_segments]
        The Rjb values for each segment
    rrup_values: array of floats
        shape: [n_segments]
        The Rrup values for each segment
    rx_values: array of floats
        shape: [n_segments]
        The Rx values for each segment
    ry_values: array of floats
        shape: [n_segments]
        The Ry values for each segment
    ry_origins: array of floats
        shape: [2, n_segments]
        The origin used for each Ry calculation
    segment_section_ids: array of ints
        shape: [n_segments]
        Section ID for each segment
    """
    # Check if all points of each segment are coplanar
    # I.e. All four points of a segment lie on the same plane
    # This only matters if the site is directly on the fault plane
    v1 = segment_nztm_coords[1, :, :] - segment_nztm_coords[0, :, :]
    v2 = segment_nztm_coords[2, :, :] - segment_nztm_coords[0, :, :]
    n = np.cross(v1, v2, axis=0)
    n /= np.linalg.norm(n, axis=0)
    v3 = segment_nztm_coords[3, :, :] - segment_nztm_coords[0, :, :]
    v3 /= np.linalg.norm(v3, axis=0)
    if np.any(np.einsum("ij,ij->j", n, v3) > 1e-2):
        raise ValueError("Not all points of each segment are coplanar")

    # Compute distances
    rjb_values, rrup_values = compute_segment_rjb_rrup(
        segment_nztm_coords, site_nztm_coords
    )
    rx_values, ry_values, ry_origins = compute_segment_rx_ry(
        segment_nztm_coords, segment_strike, segment_strike_vec, site_nztm_coords
    )
    return (
        rjb_values,
        rrup_values,
        rx_values,
        ry_values,
        ry_origins,
    )


def compute_scenario_strike(
    trace_points: np.ndarray,
    segment_strike_vecs: np.ndarray,
    segment_trace_length: np.ndarray,
    segment_section_ids: np.ndarray,
):
    """
    Compute nominal strike across rupture scenario
    Based on Spuch et al. (2015)
    Section: Strike Discordance and Nominal Strike

    Note: this calculation is done based on the segments
    as the strike varies across segments of a rupture section

    Note II: As this calculation potentially flips the segment strike
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
    unique_trace_points = np.unique(trace_points.transpose((0, 2, 1)).reshape((-1, 2)), axis=0)

    # Compute the distance matrix
    dist_matrix = np.zeros((unique_trace_points.shape[0], unique_trace_points.shape[0]))
    for i in range(unique_trace_points.shape[0]):
        dist_matrix[i, :] = np.linalg.norm(unique_trace_points[i] - unique_trace_points, axis=1)

    # Find the trace point combination that has the maximum separation distance
    ix_1, ix_2 = np.unravel_index(dist_matrix.argmax(), dist_matrix.shape)

    # Compute possible vectors
    v1 = unique_trace_points[ix_1] - unique_trace_points[ix_2]
    v1 /= np.linalg.norm(v1)
    v2 = unique_trace_points[ix_2] - unique_trace_points[ix_1]
    v2 /= np.linalg.norm(v2)

    # Choose the east pointing one and compute a_hat
    a = v1 if v1[0] > 0 else v2
    a_hat = a / np.linalg.norm(a)

    # Compute the "strike" per section/fault trace
    # based on the equation for e_j in Spuch et al. (2015)
    # I.e. the vector from the origin to the end of the trace
    # Todo: Is there a better way to do this??
    _, unique_ind = np.unique(segment_section_ids, return_index=True)
    section_strike_vecs = np.zeros((2, unique_ind.size))
    for i, ix in enumerate(unique_ind):
        cur_section_id = segment_section_ids[ix]
        m = segment_section_ids == cur_section_id
        # Compute the two possible strike vectors
        v1 = trace_points[:, :, m][0, :, 0] - trace_points[:, :, m][1, :, -1]
        v1 /= np.linalg.norm(v1)
        v2 = trace_points[:, :, m][1, :, -1] - trace_points[:, :, m][0, :, 0]
        v2 /= np.linalg.norm(v2)

        # Compute the average segment strike vector
        avg_segment_strike_vec = (segment_strike_vecs[:, m] * segment_trace_length[m]).sum(axis=1)
        avg_segment_strike_vec /= segment_trace_length[m].sum()

        # Choose the correct section strike vector
        if np.dot(v1, avg_segment_strike_vec) > np.dot(v2, avg_segment_strike_vec):
            section_strike_vecs[:, i] = v1
        else:
            section_strike_vecs[:, i] = v2

    # Compute e_j = strike_vec . a_hat
    e_j = np.matmul(segment_strike_vecs.T, a_hat)
    # e_j = np.einsum("ij,i->j", section_strike_vecs, a_hat)

    # Compute E
    E = np.sum(e_j)

    # Switch any strike vectors with opposite sign to E
    if np.any((segment_strike_flip_mask := np.sign(e_j) != np.sign(E))):
        segment_strike_vecs = segment_strike_vecs.copy()
        segment_strike_vecs[:, segment_strike_flip_mask] = (
            -1.0 * segment_strike_vecs[:, segment_strike_flip_mask]
        )

    # Compute nominal strike
    scenario_strike_vec = np.sum(segment_trace_length * segment_strike_vecs, axis=1)
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
        segment_strike_flip_mask,
    )


# def indef_integral_omega(u: float, A: np.ndarray, B: np.ndarray, S: np.ndarray):
#     """
#     Indefinite integral of segment weight function
#     https://www.wolframalpha.com/input?i=integral+of+1%2F%28%28A1+%2B+u+*+B1+-+u+*+A1+-+S1%29%5E2+%2B+%28+A2+%2B+u+*+B2+-+u+*+A2+-+S2%29%5E2%29+wrt+u
#     """
#     denominator = (
#         A[0] * (S[1] - B[1]) + A[1] * (B[0] - S[0]) - B[0] * S[1] + B[1] * S[0]
#     )
#     tan_numerator = (
#         A[0] ** 2 * (u - 1.0)
#         + A[0] * (-2 * B[0] * u + B[0] + S[0])
#         + A[1] ** 2 * (u - 1.0)
#         + A[1] * (-2 * B[1] * u + B[1] + S[1])
#         + B[0] ** 2 * u
#         - B[0] * S[0]
#         + B[1] ** 2 * u
#         - B[1] * S[1]
#     )
#     return np.arctan(tan_numerator / denominator) / denominator


def compute_segment_weight(
    segment_ry: np.ndarray, segment_rx: np.ndarray, segment_trace_length: np.ndarray
):
    """
    Computes the normalized segment weights using
    equation 4 and 5 from Spudich et al. (2015)
    """
    sw = (
        np.arctan((segment_trace_length - segment_ry) / segment_rx)
        - np.arctan(-segment_ry / segment_rx)
    ) / segment_rx

    # Site is on the extension of the segment
    if np.any(
        m := np.isclose(segment_rx, 0)
        & ((segment_ry < 0) | (segment_ry > segment_trace_length))
    ):
        sw[m] = 1 / (segment_ry[m] - segment_trace_length[m]) - 1 / segment_rx[m]

    return sw / np.sum(sw)


def compute_scenario_rx_ry(
    section_ids: np.ndarray,
    segment_nztm_coords: np.ndarray,
    segment_strike_vec: np.ndarray,
    segment_trace_length: np.ndarray,
    segment_section_ids: np.ndarray,
    segment_rjb: np.ndarray,
    segment_rx: np.ndarray,
    segment_ry: np.ndarray,
    segment_ry_origin: np.ndarray,
):
    # Get scenario data
    scenario_segment_mask = np.isin(segment_section_ids, section_ids)
    trace_points = segment_nztm_coords[::2, :2, scenario_segment_mask]
    segment_strike_vecs = segment_strike_vec[:, scenario_segment_mask]
    segment_trace_length = segment_trace_length[scenario_segment_mask]
    segment_rx = segment_rx[scenario_segment_mask].copy()
    segment_ry = segment_ry[scenario_segment_mask].copy()
    segment_origins = segment_ry_origin[:, scenario_segment_mask].copy()

    # Ignore anything outside of distance boundaries
    rjb = np.min(segment_rjb[scenario_segment_mask])
    if rjb > 500:
        return None, None

    # Compute scenario strike
    (
        scenario_strike_vec,
        _,
        scenario_origin,
        segment_strike_flip_mask,
    ) = compute_scenario_strike(trace_points, segment_strike_vecs, segment_trace_length, segment_section_ids)

    # Change sign of the Rx values corresponding to the segments
    # with a flipped strike
    segment_rx[segment_strike_flip_mask] *= -1

    # Compute the segment weights
    segment_weights = compute_segment_weight(
        segment_ry, segment_rx, segment_trace_length
    )

    ### Compute T
    T = np.average(segment_rx, weights=segment_weights)

    # If any of the segment T values are zero, set scenario T to zero
    if np.any(m := segment_rx == 0):
        T[m] = 0

    ### Compute U
    ## Update the segment origins for segments with flipped strike
    segment_origins[:, segment_strike_flip_mask] = (
        segment_origins[:, segment_strike_flip_mask]
        + segment_trace_length[segment_strike_flip_mask]
        * segment_strike_vecs[:, segment_strike_flip_mask]
        * 1e3
    )

    # Vector between scenario origin and each segment origin
    scenario_origin_segment_origin_vec = (
        segment_origins - scenario_origin[:, np.newaxis]
    )

    # Compute the order of the segments
    # based on distance from origin along
    # nominal strike
    # (segment origin - scenario_origin) . scenario_strike_vec
    segments_origin_strike_dist = (
        np.einsum("ij, i->j", scenario_origin_segment_origin_vec, scenario_strike_vec)
        / 1e3
    )

    # Origin shift has to be computed per section of the rupture scenario
    segments_origin_shift = np.full(
        segments_origin_strike_dist.shape, fill_value=np.nan
    )
    for cur_section_id in section_ids:
        # Get the segments for the current section and
        # sort these based on distance from the origin
        section_mask = segment_section_ids[scenario_segment_mask] == cur_section_id
        sort_ind = segments_origin_strike_dist[section_mask].argsort()

        # Compute the origin shift for this section
        cur_section_origin_shift = segments_origin_strike_dist[section_mask][
            sort_ind[0]
        ]

        # Compute the shift for each segment in this section
        segments_origin_shift[section_mask] = cur_section_origin_shift + np.concatenate(
            (
                [0],
                np.cumsum(segment_trace_length[section_mask][sort_ind][:-1]),
            )
        )

        # Return to original segment order
        # by re-applying sort index
        segments_origin_shift[section_mask] = segments_origin_shift[section_mask][
            sort_ind
        ]

    # Flip sign if strike was changed
    # TODO: Also need to update the origin??!!
    segment_ry[segment_strike_flip_mask] = (
        -1 * segment_ry[segment_strike_flip_mask]
        + segment_trace_length[segment_strike_flip_mask]
    )

    # Compute U
    U = np.average(
        segment_ry + segments_origin_shift,
        weights=segment_weights,
    )
    return T, U