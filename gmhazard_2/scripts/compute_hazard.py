import time
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from gmhazard_2 import gmm_logic_tree
from gmhazard_2 import source_model
from gmhazard_2 import dbs
from gmhazard_2 import im
from gmhazard_2 import hazard
from gmhazard_2 import constants

gmm_lt_ffp = Path(
    "/home/claudy/dev/work/data/nshm/resources/NZ_NSHM_GMM_LT_final_EE.xml"
)
source_model_lt_ffp = Path(
    "/home/claudy/dev/work/data/nshm/resources/source_branches.json"
)
source_db_ffp = Path("/home/claudy/dev/work/data/gmhazard_2/source_db.hdf5")

gmm_results_ffp = Path("/home/claudy/dev/work/tmp/gm_hazard_2/gmm_values.pkl")

# Get rupture sections and rupture scenarios
start_time = time.time()
with dbs.SourceModelDB(source_db_ffp) as db:
    rupture_scenarios_df = db.get_fault_rupture_scenarios()
print(f"Took {time.time() - start_time} to load the data")

start_time = time.time()
with open(gmm_results_ffp, "rb") as f:
    gmm_results = pickle.load(f)
print(f"Took {time.time() - start_time} to load the GMM results")

# Parse the GMM logic tree
start_time = time.time()
gmm_lt = gmm_logic_tree.parse_nshm_gmm_lt(gmm_lt_ffp)
print(f"Took {time.time() - start_time} to parse the GMM LT")

# Get the source model logic tree
start_time = time.time()
sm_lt = source_model.parse_nshm_source_lt(source_model_lt_ffp, source_db_ffp)
print(f"Took {time.time() - start_time} to parse the source model LT")

## TODO: Compute the hazard across all source model branches at once
# Might need to add distance filter for this to keep the number of scenarios reasonable

# Compute the hazard
cur_im_type = im.IMType.PGA
cur_im = im.IM(cur_im_type)
cur_im_levels = hazard.get_im_values(cur_im)


sb_hazard, sb_weights = hazard.get_source_branches_hazard(
    cur_im, rupture_scenarios_df, sm_lt, gmm_lt, gmm_results, n_procs=14
)

# Compute the mean hazard
cur_hazard = np.average(sb_hazard, axis=0, weights=sb_weights)

# Plot it
fig = plt.figure(figsize=(16, 10))

plt.loglog(cur_im_levels, cur_hazard, label="Fault")

plt.xlabel(f"IM ({cur_im_type})")
plt.ylabel(f"Exceedance Probability")
plt.grid(linewidth=0.5, alpha=0.5, linestyle="--")
plt.tight_layout()

plt.show()

print(f"wtf")

# cur_results_tect_type = np.array(cur_results_tect_type)
# cur_results = np.array(cur_results)
# cur_weights = np.array(cur_weights)

# Compute the mean hazard


print(f"wtf")
