import time
from typing import Dict
import multiprocessing as mp

import numpy as np
import pandas as pd
from scipy import stats


from .im import IM, IMType
from . import gmm_logic_tree
from . import constants
from . import source_model


def get_source_branches_hazard(
    im: IM,
    rupture_scenarios_df: pd.DataFrame,
    sm_lt: source_model.SMLogicTree,
    gmm_lt: gmm_logic_tree.GMMLogicTree,
    gmm_results: Dict[
        IMType,
        Dict[constants.TectonicType, Dict[gmm_logic_tree.GMMRunConfig, pd.DataFrame]],
    ],
    n_procs: int = 1,
):
    im_levels = get_im_values(im)

    start_time = time.time()
    if n_procs == 1:
        sb_hazard = []
        sb_weights = []
        for i, cur_sm_branch in enumerate(sm_lt.branches.values()):
            # m = np.isin(rupture_scenarios_df.set_id, cur_sm_branch.flt_source_set_ids)
            # cur_rupture_scenario_df = rupture_scenarios_df.loc[m]
            #
            # cur_hazard = compute_sb_hazard(
            #     cur_rupture_scenario_df, gmm_lt, gmm_results, im, im_levels
            # )

            cur_hazard, cur_weight = _get_sb_hazard(
                cur_sm_branch, rupture_scenarios_df, gmm_lt, gmm_results, im, im_levels
            )

            sb_hazard.append(cur_hazard)
            sb_weights.append(cur_sm_branch.weight)
    else:
        with mp.Pool(n_procs) as p:
            results = p.starmap(
                _get_sb_hazard,
                [
                    (
                        cur_sm_branch,
                        rupture_scenarios_df,
                        gmm_lt,
                        gmm_results,
                        im,
                        im_levels,
                    )
                    for cur_sm_branch in sm_lt.branches.values()
                ],
            )
        sb_hazard, sb_weights = zip(*results)
    print(f"Time taken: {time.time() - start_time:.2f}s")

    return np.array(sb_hazard), np.array(sb_weights)


def _get_sb_hazard(
    sm_branch: source_model.SMLogicTreeBranch,
    rupture_scenarios_df: pd.DataFrame,
    gmm_lt: gmm_logic_tree.GMMLogicTree,
    gmm_results: Dict[
        IMType,
        Dict[constants.TectonicType, Dict[gmm_logic_tree.GMMRunConfig, pd.DataFrame]],
    ],
    im: IM,
    im_levels: np.ndarray,
):
    m = np.isin(rupture_scenarios_df.set_id, sm_branch.flt_source_set_ids)
    cur_rupture_scenario_df = rupture_scenarios_df.loc[m]

    hazard = compute_sb_hazard(
        cur_rupture_scenario_df, gmm_lt, gmm_results, im, im_levels
    )

    return hazard, sm_branch.weight


def compute_sb_hazard(
    rupture_scenario_df: pd.DataFrame,
    gmm_lt: gmm_logic_tree.GMMLogicTree,
    gmm_results: Dict[
        IMType,
        Dict[constants.TectonicType, Dict[gmm_logic_tree.GMMRunConfig, pd.DataFrame]],
    ],
    im: IM,
    im_levels: np.ndarray,
):
    # Get the list of rupture scenario ids
    scenario_ids = rupture_scenario_df.index.values
    rupture_scenario_tect_types = np.unique(
        rupture_scenario_df.tect_type.values
    ).astype(str)

    # Compute the hazard for each GMM "branch"
    # Note: These are not full LT branches as they only
    # contain results for a single tectonic type
    gmm_lt_tect_types = list(gmm_lt.branches.keys())
    gmm_hazard_results = {cur_tect_type: [] for cur_tect_type in gmm_lt_tect_types}
    for cur_tect_type, cur_lt in gmm_lt.branches.items():
        for cur_branch in cur_lt.branches.values():
            # No ruptures for current tectonic type
            ### TODO: Fix this, currently this is incorrect (I think)!!
            if cur_tect_type.value not in rupture_scenario_tect_types:
                gmm_hazard_results[cur_tect_type].append(np.zeros_like(im_levels))
                continue

            # Get the GMM values for the current "branch"
            gmm_values = gmm_results[im.im_type][cur_tect_type][cur_branch.run_config]
            gmm_values = gmm_values.loc[np.isin(gmm_values.index.values, scenario_ids)]

            # Compute the rupture exceedance probabilities
            rupture_exd_prob = stats.norm.sf(
                np.log(im_levels).reshape(1, -1),
                loc=gmm_values.mu.values.reshape(-1, 1),
                scale=gmm_values.sigma.values.reshape(-1, 1),
            )

            # Compute the hazard
            excd_prob = cur_branch.weight * np.sum(
                rupture_exd_prob
                * rupture_scenario_df.loc[gmm_values.index, "prob_occur"].values[
                    :, np.newaxis
                ],
                axis=0,
            )

            gmm_hazard_results[cur_tect_type].append(excd_prob)

    # Combine the results
    for cur_tect_type in gmm_hazard_results.keys():
        gmm_hazard_results[cur_tect_type] = np.array(gmm_hazard_results[cur_tect_type])

    ### Compute all logic tree branches,
    ### i.e. all combinations of the tectonic type GMMs
    ###   and sum to compute the hazard
    # Only currently works for these 3 tectonic types
    assert len(gmm_hazard_results.keys()) == 3
    sb_hazard = np.sum(
        gmm_hazard_results[constants.TectonicType.active_shallow][
            :, np.newaxis, np.newaxis, :
        ]
        + gmm_hazard_results[constants.TectonicType.subduction_slab][
            np.newaxis, :, np.newaxis, :
        ]
        + gmm_hazard_results[constants.TectonicType.subduction_interface][
            np.newaxis, np.newaxis, :, :
        ],
        axis=(0, 1, 2),
    )

    return sb_hazard


def get_min_max_values_for_im(im: IM):
    """Get minimum and maximum for the given im. Values for velocity are
    given on cm/s, acceleration on cm/s^2 and Ds on s
    """
    if im.is_pSA():
        assert im.period is not None, "No period provided for pSA, this is an error"
        if im.period <= 0.5:
            return 0.005, 10.0
        elif 0.5 < im.period <= 1.0:
            return 0.005, 7.5
        elif 1.0 < im.period <= 3.0:
            return 0.0005, 5.0
        elif 3.0 < im.period <= 5.0:
            return 0.0005, 4.0
        elif 5.0 < im.period <= 10.0:
            return 0.0005, 3.0
    if im.im_type is IMType.PGA:
        return 0.0001, 10.0
    elif im.im_type is IMType.PGV:
        return 1.0, 400.0
    elif im.im_type is IMType.CAV:
        return 0.0001 * 980, 20.0 * 980.0
    elif im.im_type is IMType.AI:
        return 0.01, 1000.0
    elif im.im_type is IMType.Ds575 or im.im_type is IMType.Ds595:
        return 1.0, 400.0
    else:
        print("Unknown IM, cannot generate a range of IM values. Exiting the program")
        exit(1)


def get_im_values(im: IM, n_values: int = 100):
    """
    Create an range of values for a given IM according to their min, max
    as defined by get_min_max_values

    Parameters
    ----------
    im: IM
        The IM Object to get im values for
    n_values: int

    Returns
    -------
    Array of IM values
    """
    start, end = get_min_max_values_for_im(im)
    im_values = np.logspace(
        start=np.log(start), stop=np.log(end), num=n_values, base=np.e
    )
    return im_values
