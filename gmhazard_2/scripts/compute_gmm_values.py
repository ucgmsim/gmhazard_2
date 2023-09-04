import pickle
import time
from numba.typed import List
from pathlib import Path
import multiprocessing as mp

import numba as nb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from qcore import geo

from gmhazard_2 import dbs
from gmhazard_2 import distance
from gmhazard_2 import gmm_logic_tree
from gmhazard_2 import source_model
from gmhazard_2 import source
from gmhazard_2 import constants
from gmhazard_2 import utils
from gmhazard_2 import im
from empirical.util.openquake_wrapper_vectorized import oq_run
from empirical.util.classdef import TectType

# source_db_ffp = Path("/Users/claudy/dev/work/tmp/nshm/source_db.hdf5")
# gmm_lt_ffp = Path("/Users/claudy/dev/work/data/nshm/resources/NZ_NSHM_GMM_LT_final_EE.xml")
# source_model_lt_ffp = Path("/Users/claudy/dev/work/data/nshm/resources/source_branches.json")

n_procs = 16
source_db_ffp = Path("/home/claudy/dev/work/data/gmhazard_2/source_db.hdf5")
gmm_lt_ffp = Path(
    "/home/claudy/dev/work/data/nshm/resources/NZ_NSHM_GMM_LT_final_EE.xml"
)
source_model_lt_ffp = Path(
    "/home/claudy/dev/work/data/nshm/resources/source_branches.json"
)
backarc_ffp = Path("/home/claudy/dev/work/data/nshm/resources/backarc.json")

output_ffp = Path("/home/claudy/dev/work/tmp/gm_hazard_2/gmm_values.pkl")


IM_types = [im.IMType.PGA, im.IMType.pSA]
periods = constants.SA_PERIODS


# site_coords = np.asarray((172.63, -43.53))
site_coords = np.asarray((168.2, -45.5))
site_nztm_coords = distance.site_to_nztm(site_coords)

site_vs30 = 760
site_z1p0 = 0.5
site_z2p5 = 2.0

site_back = utils.get_backarc_mask(backarc_ffp, site_coords[np.newaxis, :])[0]

# Parse the GMM logic tree
start_time = time.time()
gmm_lt = gmm_logic_tree.parse_nshm_gmm_lt(gmm_lt_ffp)
print(f"Took {time.time() - start_time} to parse the GMM LT")

# Get the list of GMMs utilised
start_time = time.time()
gmm_run_configs = gmm_logic_tree.get_tec_type_gmm_run_configs(gmm_lt)
print(f"Took {time.time() - start_time} to get the GMM run configs")

# Get the source model logic tree
start_time = time.time()
sm_lt = source_model.parse_nshm_source_lt(source_model_lt_ffp, source_db_ffp)
print(f"Took {time.time() - start_time} to parse the source model LT")

# Get rupture sections and rupture scenarios
start_time = time.time()
with dbs.SourceModelDB(source_db_ffp) as db:
    source_set_info = db.get_source_set_info()
    rupture_section_pts_df = db.get_flt_rupture_section_pts()
    rupture_scenarios_df = db.get_fault_rupture_scenarios()
print(f"Took {time.time() - start_time} to load the data")

# Get the segment coords
start_time = time.time()
segment_coords, segment_section_ids = distance.get_segment_coords(
    rupture_section_pts_df
)
segment_nztm_coords = distance.segment_to_nztm(segment_coords)


start_time = time.time()
distance_df = distance.get_scenario_distances(
    rupture_scenarios_df, segment_nztm_coords, segment_section_ids, site_nztm_coords
)
print(f"Took {time.time() - start_time} to compute the distances")


start_time = time.time()
source_df = source.get_scenario_source_props(
    rupture_scenarios_df, segment_nztm_coords, segment_section_ids
)
print(f"Took {time.time() - start_time} to compute the source properties")

# Create the rupture df
rupture_df = pd.merge(distance_df, source_df, right_index=True, left_index=True)
rupture_df["vs30"] = site_vs30
rupture_df["z1pt0"] = site_z1p0
rupture_df["z2pt5"] = site_z2p5
rupture_df["vs30measured"] = False
rupture_df["backarc"] = site_back

NSHM_TECTONIC_TYPE_MAPPING = {
    "Active Shallow Crust": constants.TectonicType.active_shallow.value,
    "Subduction Interface": constants.TectonicType.subduction_interface.value,
    "Subduction Intraslab": constants.TectonicType.subduction_slab.value,
}

for cur_t_type in rupture_scenarios_df.tect_type.unique():
    m = rupture_scenarios_df.tect_type == cur_t_type
    rupture_df.loc[m, "tect_type"] = NSHM_TECTONIC_TYPE_MAPPING[str(cur_t_type)]

rupture_df["mag"] = rupture_scenarios_df.mag.values
rupture_df["rake"] = rupture_scenarios_df.rake.values

EMPIRICAL_ENGINE_TEC_TYPE_MAPPING = {
    constants.TectonicType.active_shallow: TectType.ACTIVE_SHALLOW,
    constants.TectonicType.subduction_interface: TectType.SUBDUCTION_INTERFACE,
    constants.TectonicType.subduction_slab: TectType.SUBDUCTION_SLAB,
}


def compute_results(
    tect_type: constants.TectonicType,
    run_config: gmm_logic_tree.GMMRunConfig,
    rupture_df: pd.DataFrame,
    im_type: im.IMType,
    periods: np.ndarray = None,
):
    kwargs = {} if run_config.gmm_parameters is None else run_config.gmm_parameters
    start_time = time.time()
    cur_rupture_df = rupture_df.loc[rupture_df.tect_type == tect_type.value]
    if cur_rupture_df.shape[0] == 0:
        return None
    else:
        results = oq_run(
            run_config.gmm,
            EMPIRICAL_ENGINE_TEC_TYPE_MAPPING[tect_type],
            cur_rupture_df,
            str(im_type),
            periods=periods,
            **kwargs,
        )
        print(
            f"{run_config.gmm} - {run_config.gmm_parameters} - {tect_type} "
            f"- {im_type}: {time.time() - start_time}"
        )

        results.index = cur_rupture_df.index
        if im_type is not im_type.pSA:
            results = results.iloc[:, :2]
            results.columns = ["mu", "sigma"]
        else:
            cols = [
                cur_col
                for cur_col in results.columns
                if "mean" in cur_col or "std_Total" in cur_col
            ]
            results = results[cols]
            results.columns = [
                cur_col.replace("mean", "mu").replace("std_Total", "sigma")
                for cur_col in cols
            ]
        return results


# start_time = time.time()
# gmm_results = {}
# for cur_im in IMs:
#     gmm_results[cur_im] = {}
#     for cur_tec_type in gmm_run_configs.keys():
#         gmm_results[cur_im][cur_tec_type] = {}
#         for cur_run_config in gmm_run_configs[cur_tec_type]:
#             # print(f"Processing - {cur_tec_type} - {cur_run_config}")
#             gmm_results[cur_im][cur_tec_type][hash(cur_run_config)] = compute_results(
#                 cur_tec_type,
#                 cur_run_config,
#                 rupture_df,
#                 cur_im,
#                 periods if cur_im == "pSA" else None,
#             )
# print(f"Took {time.time() - start_time} to compute GMM results")

# Create the work chunks
work_chunks = []
for cur_im_type in IM_types:
    for cur_tect_type in gmm_run_configs.keys():
        for cur_run_config in gmm_run_configs[cur_tect_type]:
            work_chunks.append(
                (
                    cur_tect_type,
                    cur_run_config,
                    rupture_df,
                    cur_im_type,
                    periods if cur_im_type is im.IMType.pSA else None,
                )
            )

start_time = time.time()
with mp.Pool(n_procs) as p:
    results = p.starmap(compute_results, work_chunks)
print(f"Took {time.time() - start_time} to compute GMM results")

# Extract the results
# Keys IM, TectType, RunConfig
gmm_results = {}
for cur_result, cur_run_details in zip(results, work_chunks):
    if cur_result is None:
        continue

    if (cur_im := cur_run_details[-2]) not in gmm_results.keys():
        gmm_results[cur_im] = {}

    if (cur_tect_type := cur_run_details[0]) not in gmm_results[cur_im].keys():
        gmm_results[cur_im][cur_tect_type] = {}

    gmm_results[cur_im][cur_tect_type][cur_run_details[1]] = cur_result

with output_ffp.open("wb") as f:
    pickle.dump(gmm_results, f)


print(f"wtf")



# start_time = time.time()
# gmm_results = {}
# for cur_im in IMs:
#     for cur_tec_type in gmm_run_configs.keys():
#         gmm_results[cur_tec_type] = {}
#         for cur_run_config in gmm_run_configs[cur_tec_type]:
#             print(f"Processing - {cur_tec_type} - {cur_run_config}")
#             gmm_results[cur_tec_type][hash(cur_run_config)] = compute_results(
#                 cur_tec_type,
#                 cur_run_config,
#                 rupture_df,
#                 cur_im,
#                 periods if cur_im == "pSA" else None,
#             )
# print(f"Took {time.time() - start_time} to compute GMM results")


# cur_tect_type = list(gmm_run_configs.keys())[0]
# r = oq_run(
#     gmm_run_configs[cur_tect_type][-1].gmm, TectType.ACTIVE_SHALLOW, rupture_df, "PGA"
# )
